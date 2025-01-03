import random
import ssl
import websockets
import asyncio
import os
import sys
import json
import argparse
import traceback

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
gi.require_version('GstWebRTC', '1.0')
from gi.repository import GstWebRTC
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp

PIPELINE_DESC = '''
webrtcbin name=sendrecv stun-server=stun://stun.l.google.com:19302
 udpsrc port=10000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T" name=udpsrc0 ! 
 rtpjitterbuffer mode=0 latency=0 ! queue max-size-buffers=4096 !
 rtpmp2tdepay ! tsdemux ! h264parse ! decodebin name=decoder !
 tee name=t ! queue ! videoconvert ! videoscale ! video/x-raw,format=I420 !
 x264enc tune=zerolatency speed-preset=ultrafast bitrate=1000 ! 
 rtph264pay config-interval=-1 timestamp-offset=0 !
 queue ! application/x-rtp,media=video,encoding-name=H264,payload=98 ! sendrecv.
 t. ! queue ! fakesink sync=false name=fakesink0
'''

class WebRTCClient:
    def __init__(self, id_, peer_id, server):
        self.id_ = id_
        self.conn = None
        self.pipe = None
        self.webrtc = None
        self.peer_id = peer_id
        self.server = server or 'ws://prism:80/audio_reactive/webrtc'
        # self.server = server or 'ws://prism:80/audio_reactive/webrtc'

    async def connect(self):
        self.conn = await websockets.connect(self.server)
        await self.conn.send(json.dumps({
            "type": "hello",
            "id": self.id_
        }))

    async def setup_call(self):
        await self.conn.send(json.dumps({
            "type": "session",
            "peer_id": self.peer_id
        }))

    def send_sdp_offer(self, offer):
        text = offer.sdp.as_text()
        print('Sending offer:\n%s' % text)
        msg = json.dumps({
            "type": "sdp",
            "sdp": {
                "type": "offer",
                "sdp": text
            }
        })
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.conn.send(msg))

    def on_offer_created(self, promise, _, __):
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value('offer')
        promise = Gst.Promise.new()
        self.webrtc.emit('set-local-description', offer, promise)
        promise.interrupt()
        self.send_sdp_offer(offer)

    def on_negotiation_needed(self, element):
        promise = Gst.Promise.new_with_change_func(self.on_offer_created, element, None)
        element.emit('create-offer', None, promise)

    def send_ice_candidate_message(self, _, mlineindex, candidate):
        icemsg = json.dumps({
            'type': 'ice',
            'ice': {
                'candidate': candidate,
                'sdpMLineIndex': mlineindex
            }
        })
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.conn.send(icemsg))

    def on_incoming_stream(self, _, pad):
        print(f"New pad: {pad.get_name()} for {pad.get_parent().get_name()}", flush=True)
        if pad.direction != Gst.PadDirection.SRC:
            return
            
        print("Creating video processing pipeline...", flush=True)
        # Create a video converter and encoder pipeline
        queue = Gst.ElementFactory.make('queue')
        convert = Gst.ElementFactory.make('videoconvert')
        encoder = Gst.ElementFactory.make('x264enc')
        payloader = Gst.ElementFactory.make('rtph264pay')
        
        # Add elements to pipeline
        for element in [queue, convert, encoder, payloader]:
            if not element:
                print(f"Failed to create element {element}", flush=True)
                return
            self.pipe.add(element)
            element.sync_state_with_parent()
        
        # Link elements
        if not pad.link(queue.get_static_pad('sink')):
            print("Failed to link pad to queue", flush=True)
            return
        if not queue.link(convert):
            print("Failed to link queue to convert", flush=True)
            return
        if not convert.link(encoder):
            print("Failed to link convert to encoder", flush=True)
            return
        if not encoder.link(payloader):
            print("Failed to link encoder to payloader", flush=True)
            return
        
        print("Video processing pipeline created successfully", flush=True)
        
        # Connect to the payloader's src pad to get encoded video
        payloader.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_video_data
        )

    def on_video_data(self, pad, info):
        buffer = info.get_buffer()
        if buffer and self.conn:
            print(f"Received video buffer of size: {buffer.get_size()}", flush=True)
            try:
                data = buffer.extract_dup(0, buffer.get_size())
                # Send video data through websocket
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.conn.send(json.dumps({
                    "type": "video",
                    "data": data.hex()  # Convert binary data to hex string
                })))
            except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                print(f"Connection is closed: {str(e)}", flush=True)
            except Exception as e:
                print(f"Error sending video data: {str(e)}", flush=True)
        return Gst.PadProbeReturn.OK

    def start_pipeline(self):
        print("Starting pipeline...", flush=True)
        self.pipe = Gst.parse_launch(PIPELINE_DESC)
        
        # Add message handling
        bus = self.pipe.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_message)
        
        # Add probes to track data flow
        udpsrc = self.pipe.get_by_name('udpsrc0')
        if udpsrc:
            pad = udpsrc.get_static_pad('src')
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.on_udp_data)
                print("Added probe to UDP source", flush=True)
        
        decoder = self.pipe.get_by_name('decoder')
        if decoder:
            decoder.connect('pad-added', self.on_decoder_pad)
            print("Connected to decoder pad-added signal", flush=True)
        
        self.webrtc = self.pipe.get_by_name('sendrecv')
        self.webrtc.connect('on-negotiation-needed', self.on_negotiation_needed)
        self.webrtc.connect('on-ice-candidate', self.send_ice_candidate_message)
        self.webrtc.connect('pad-added', self.on_incoming_stream)
        
        ret = self.pipe.set_state(Gst.State.PLAYING)
        print(f"Pipeline set to PLAYING: {ret}", flush=True)

    def on_udp_data(self, pad, info):
        if info.get_buffer():
            print("Received UDP data", flush=True)
        return Gst.PadProbeReturn.OK

    def on_decoder_pad(self, element, pad):
        print(f"Decoder pad added: {pad.get_name()}", flush=True)
        if pad.get_name().startswith('src'):
            print("Found video source pad from decoder", flush=True)
            caps = pad.get_current_caps()
            if caps:
                structure = caps.get_structure(0)
                print(f"Video caps: {structure.to_string()}", flush=True)

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error: {err.message}", flush=True)
            print(f"Debug info: {debug}", flush=True)
            print(f"Error source: {message.src.get_name()}", flush=True)
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipe:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick} (pending: {pending_state.value_nick})", flush=True)
        elif t == Gst.MessageType.EOS:
            print("End of stream", flush=True)
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Pipeline warning: {warn.message}", flush=True)
            print(f"Debug info: {debug}", flush=True)
            print(f"Warning source: {message.src.get_name()}", flush=True)

    async def handle_sdp(self, message):
        assert (self.webrtc)
        msg = json.loads(message)
        if msg.get('type') == 'sdp_response' and 'sdp' in msg:
            sdp = msg['sdp']
            if not isinstance(sdp, dict) or 'type' not in sdp or 'sdp' not in sdp:
                print("Invalid SDP format received", flush=True)
                return
                
            print(f"Processing SDP {sdp['type']}", flush=True)
            sdp_str = sdp['sdp']
            print('Received SDP:\n%s' % sdp_str)
            
            res, sdpmsg = GstSdp.SDPMessage.new()
            GstSdp.sdp_message_parse_buffer(bytes(sdp_str.encode()), sdpmsg)
            answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg)
            promise = Gst.Promise.new()
            self.webrtc.emit('set-remote-description', answer, promise)
            promise.interrupt()
            
        elif msg.get('type') == 'ice_response' and 'ice' in msg:
            ice = msg['ice']
            if not isinstance(ice, dict) or 'candidate' not in ice or 'sdpMLineIndex' not in ice:
                print("Invalid ICE format received", flush=True)
                return
                
            candidate = ice['candidate']
            sdpmlineindex = ice['sdpMLineIndex']
            print(f"Adding ICE candidate: {candidate}", flush=True)
            self.webrtc.emit('add-ice-candidate', sdpmlineindex, candidate)
        else:
            print(f"Unhandled message type: {msg.get('type')}", flush=True)

    async def close(self):
        if self.pipe:
            self.pipe.set_state(Gst.State.NULL)
        if self.conn:
            try:
                await self.conn.close()
            except Exception as e:
                print(f"Error closing connection: {str(e)}", flush=True)

    async def loop(self):
        assert self.conn
        try:
            # Start pipeline immediately after connection
            print("Starting pipeline immediately...", flush=True)
            self.start_pipeline()
            
            async for message in self.conn:
                try:
                    msg = json.loads(message)
                    msg_type = msg.get('type', '')
                    print(f"Received message type: {msg_type}", flush=True)
                    
                    if msg_type == 'video_received':
                        print(f"Server received video data of size: {msg.get('size')} bytes", flush=True)
                    elif msg_type == 'sdp_response':
                        await self.handle_sdp(json.dumps(msg))
                    elif msg_type == 'ice_response':
                        await self.handle_sdp(json.dumps(msg))
                    elif msg_type == 'error':
                        print(f"Error from server: {msg.get('message')}", flush=True)
                        return 1
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}", flush=True)
                except Exception as e:
                    print(f"Error processing message: {str(e)}", flush=True)
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed", flush=True)
        except Exception as e:
            print(f"Loop error: {str(e)}", flush=True)
            traceback.print_exc()
        finally:
            await self.close()
        return 0


def check_plugins():
    needed = ["opus", "vpx", "nice", "webrtc", "dtls", "srtp", "rtp",
              "rtpmanager", "videotestsrc", "audiotestsrc"]
    missing = list(filter(lambda p: Gst.Registry.get().find_plugin(p) is None, needed))
    if len(missing):
        print('Missing gstreamer plugins:', missing)
        return False
    return True


if __name__=='__main__':
    Gst.init(None)
    if not check_plugins():
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('peerid', help='String ID of the peer to connect to')
    parser.add_argument('--server', help='Signalling server to connect to, eg "wss://127.0.0.1:8443"')
    args = parser.parse_args()
    our_id = random.randrange(10, 10000)
    c = WebRTCClient(our_id, args.peerid, args.server)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(c.connect())
        loop.run_until_complete(c.loop())
    except KeyboardInterrupt:
        print("Interrupted by user", flush=True)
    finally:
        loop.run_until_complete(c.close())
        loop.close()
    sys.exit(0)

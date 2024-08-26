import argparse
import configparser
import asyncio
import json
import os
import platform
import ssl
import logging
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender

ROOT = os.path.dirname(__file__)
config = configparser.ConfigParser()
config.read("config/serverconfig.ini")
HOST = config["DEFAULT"]["server_host"]
PORT = int(config["DEFAULT"]["server_port"])
CAMERA_ID = config["DEFAULT"]["camera_id"]
relay = None
webcam = None


def create_local_tracks(play_from, decode):
    global relay, webcam

    if play_from:
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    else:
        # Adjust video_size to a lower resolution
        options = {
            "framerate": "25",
            "video_size": "320x240",  # Lower resolution
            "rtbufsize": "32020000",  # Increase buffer size to 7MB (you can adjust this value)
        }
        if relay is None:
            if platform.system() == "Darwin":
                camera_format = config["DEFAULT"]["mac_camera_format"]
                webcam = MediaPlayer(
                    "default:none", format=camera_format, options=options
                )
            elif platform.system() == "Windows":
                camera_format = config["DEFAULT"]["win_camera_format"]
                webcam = MediaPlayer(
                    f"video={CAMERA_ID}", format=camera_format, options=options
                )
            else:
                camera_format = config["DEFAULT"]["lin_camera_format"]
                webcam = MediaPlayer(
                    f"/dev/{CAMERA_ID}", format=camera_format, options=options
                )
            relay = MediaRelay()
        return None, relay.subscribe(webcam.video)


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            if msg.data == "handshake":
                await ws.send_str("handshake_ok")
            else:
                # Handle other messages if needed
                pass
        elif msg.type == web.WSMsgType.ERROR:
            print("WebSocket connection closed with exception %s" % ws.exception())

    return ws


async def offer(request):
    params = await request.json()

    # Validate type
    valid_types = ["offer", "pranswer", "answer", "rollback"]
    if params["type"] not in valid_types:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"error": "Invalid type"}),
        )

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
            pcs.clear()

    # Open media source
    audio, video = create_local_tracks(
        args.play_from, decode=not args.play_without_decoding
    )

    if audio:
        audio_sender = pc.addTrack(audio)
        if args.audio_codec:
            force_codec(pc, audio_sender, args.audio_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the audio codec using --audio-codec")

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def stop_server(request):
    logging.info("Stopping camera...")

    # for pc in pcs:
    #     for t in pc.getTransceivers():
    #         if t.kind == "video":
    # await t.sender.replaceTrack(None)  # Stop sending the video track

    if webcam and webcam.audio is None:
        # webcam.audio is None checks if the MediaPlayer is still active
        webcam.video.stop()  # Stops the media player video track if applicable

    return web.Response(text="Camera stopped", content_type="text/plain")


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"Host for HTTP server (default: {HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Port for HTTP server (default: {PORT})"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/stop", stop_server)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/offer", offer)

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

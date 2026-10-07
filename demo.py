"""Run the real annotation UI with generated schematic video; no participant data."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from scripts import aoi_projects
from scripts.aoi_horizontal import partition_polygons
from scripts.supervised_aoi import VideoSource, Session
from scripts.supervised_aoi_server import Server

ROOT=Path(__file__).resolve().parent

def prepare() -> tuple[Path,Path,list]:
    folder=ROOT/'outputs/demo';folder.mkdir(parents=True,exist_ok=True)
    video=folder/'synthetic-scene.avi';timestamps=folder/'pts.json'
    if not video.exists():
        output=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(960,540))
        if not output.isOpened():raise RuntimeError('Video writer unavailable')
        for i in range(180):
            frame=np.full((540,960,3),(35,42,39),dtype=np.uint8)
            offset=int(4*np.sin(i/25))
            cv2.rectangle(frame,(30,50+offset),(930,295+offset),(83,97,88),-1)
            for j in range(3):
                x=45+j*295
                cv2.rectangle(frame,(x,70+offset),(x+275,275+offset),(49,65,56),-1)
                cv2.rectangle(frame,(x+10,95+offset),(x+265,250+offset),(65,103,88),2)
                for k in range(5):
                    y=125+k*24+offset
                    cv2.line(frame,(x+20,y),(x+240,y),(91,134,116),1)
            cv2.rectangle(frame,(315,358+offset),(670,510+offset),(178,173,145),-1)
            for k in range(4):cv2.circle(frame,(370+k*75,425+offset),15,(62,107,130),-1)
            cv2.putText(frame,'SYNTHETIC DEMO - NO PARTICIPANT VIDEO',(44,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(180,205,190),1)
            cv2.putText(frame,f'FRAME {i:03} / {i/30:.2f}s',(40,525),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,205,190),1)
            output.write(frame)
        output.release()
        timestamps.write_text(json.dumps([i/30 for i in range(180)]))
    manifest=[dict(subject_id=1,segment_id=f'{i//2+1}.{i%2+1}',start_s=i*.5,end_s=(i+1)*.5,video_path=str(video)) for i in range(12)]
    timing=folder/'timing';timing.mkdir(exist_ok=True)
    (timing/'eligible_subject_inventory.json').write_text(json.dumps([dict(subject_id=1,eligible=True,image_count=0)]))
    (timing/'demo_segments.json').write_text(json.dumps(manifest))
    aoi_projects.TIMING=timing
    aoi_projects.VIDEO_ROOT=folder/'no-original-videos'
    return video,timestamps,manifest

def main() -> None:
    parser=argparse.ArgumentParser(description='AOI 合成视频演示')
    parser.add_argument('--port',type=int,default=18809)
    args=parser.parse_args()
    video,timestamps,manifest=prepare()
    source=VideoSource(video,timestamps)
    template=partition_polygons(source.size,295,48,0)
    session=Session(manifest,ROOT/'outputs/demo/session',source,template)
    if session._get('horizontal_mode') is not True:
        session.set_partition(True)
        session.correct(template,dict(screen=True,tablet=True))
    projects=aoi_projects.Projects(session,timestamps,ROOT/'outputs/demo/projects')
    server=Server(('127.0.0.1',args.port),session,projects)
    print(f'AOI 合成演示：http://127.0.0.1:{args.port}/?view=simple',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();server.session.close();server.session.source.close()

if __name__=='__main__':main()

"""
这里通过每帧读入的方式进行分割
"""
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 设置开始时间(单位秒)
    parser.add_argument('--start_hour', metavar='', type=int, default=0, help='set start hour')
    parser.add_argument('--start_min', metavar='', type=int, default=0, help='set start min')
    parser.add_argument('--start_second', metavar='', type=int, default=0, help='set start second')
    # 设置结束时间(单位秒)
    parser.add_argument('--end_hour', metavar='', type=int, default=0, help='set end hour')
    parser.add_argument('--end_min', metavar='', type=int, default=2, help='set end min')
    parser.add_argument('--end_second', metavar='', type=int, default=0, help='set end second')
    parser.add_argument('--video_src', metavar='', type=str,
                        default=r"D:\Desktop\Academic\data\边界\20220323 fuyang\video_20220323_1711_left.avi",
                        help='set path of the original video')
    parser.add_argument('--video_clip', metavar='', type=str,
                        default=r"D:\Desktop\Academic\data\边界\20220323 fuyang\video_20220323_1711_left_clip001.mp4",
                        help='set path of the clipped video')
    parser.add_argument('--FPS', metavar='', type=int, default=30, help='set FPS')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    START_TIME = args.start_hour * 3600 + args.start_min * 60 + args.start_second
    END_TIME = args.end_hour * 3600 + args.end_min * 60 + args.end_second

    cap = cv2.VideoCapture(args.video_src)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    videoWriter = cv2.VideoWriter(args.video_clip, fourcc, args.FPS, size)

    frameToStart = START_TIME * args.FPS  # 开始帧 = 开始时间*帧率
    frametoStop = END_TIME * args.FPS  # 结束帧 = 结束时间*帧率

    COUNT = 0
    while True:
        success, frame = cap.read()
        if success:
            COUNT += 1
            if frametoStop >= COUNT > frameToStart:  # 选取起始帧
                print('frame = ', COUNT)
                videoWriter.write(frame)
        if COUNT > frametoStop:
            break
    print('end')

"""
moviepy——可针对.mp4格式文件进行分割
"""
# from moviepy.editor import *
#
# if __name__ == '__main__':
#     video_path = r'D:\Desktop\video.mp4'
#     clip_video_path = r'D:\Desktop\video_clip.mp4'
#     clip = VideoFileClip(video_path).subclip(0, 2)
#     clip.write_videofile(clip_video_path)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:14:02 2023

@author: NIshanth Mohankumar
"""

def ar_overlay(dst_frames, src_frames, match_img, opts):
    num_frames = min(len(dst_frames), len(src_frames))
    res_frames = []

    sample_frame = src_frames[0]
    black_rows = np.where(np.sum(sample_frame[:, :, 0], axis = 1) == 0)[0]
    start_h = 0
    end_h = sample_frame.shape[0]

    for i in range(1, len(black_rows)):
        if(black_rows[i] != black_rows[i - 1] + 1):
            start_h = black_rows[i - 1]
            break

    for i in range(len(black_rows) - 2, -1, -1):
        if(black_rows[i] != black_rows[i + 1] - 1):
            end_h = black_rows[i + 1]
            break

    src_frames_cropped = center_crop(src_frames, match_img.shape[:2], (start_h, end_h))
    for i in range(0, num_frames):
        composite_img = warpImage(dst_frames[i], match_img, src_frames_cropped[i], opts)
        res_frames.append(composite_img)
        # print("Completed frame %d" % (i))

    res_frames = np.array(res_frames)

    return res_frames

if _name_ == "_main_":

    opts = get_opts()

    book_vid_path = "../data/book.mov"
    book_frames = loadVid(book_vid_path)

    src_vid_path = "../data/ar_source.mov"
    src_frames = loadVid(src_vid_path)

    cv_cover = cv2.imread("../data/cv_cover.jpg")

    res_frames = ar_overlay(book_frames, src_frames, cv_cover, opts)

    video = cv2.VideoWriter("../result/ar.avi", 0, 30, (res_frames.shape[2], res_frames.shape[1]))
    for frame in res_frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
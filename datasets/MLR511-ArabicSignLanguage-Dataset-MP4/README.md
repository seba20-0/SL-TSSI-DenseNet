# Dataset Description

This dataset consists of videos of participants signing gestures in Arabic Sign Language, captured using the Intel RealSense Depth Camera D415. 
Resource used for all signs: [https://jumla.mada.org.qa/dictionary/?lang=en](https://jumla.mada.org.qa/dictionary/?lang=en).

## Data Collection Details

- **Camera Model:** Intel RealSense D415
- **Distance from Camera to User:** Approximately 150 cm
- **Lighting Conditions:** Well-lit room with LED lights. No natural lighting.
- **Resolution:** 960 x 540 pixels
- **Frame Rate:** 15 FPS
- **Color Format:** RGB8
- **Participants:** 12 users, aged 20-24
- **Gestures:** 10 unique gestures per user
- **Repetitions:** 10 repetitions per gesture
- **Total Dataset Size:** 12 users × 10 gestures × 10 repetitions = **1,200 video samples**
- **Video Format:** `.mp4`

> **Note:** All users are right-handed, except users 01 and 02. Feel free to decide on whether or not you wish to keep this variability, or remove it by flipping the images along the vertical axis.

## User, Gesture, and Repitition IDs
- **Users** are identified as user01 -> user12
- **Gestures** are identified as G01 -> G10, and they are as follows:
    01. Hi
    02. Please
    03. What?
    04. Arabic
    05. University 
    06. You  
    07. Eat
    08. Sleep
    09. Go
    10. UAE 
- **Repititions** are identified as R01 -> R10

## File Hierarchy and Organization

The dataset is organized as follows:

```
user01/
    G01/
        R01/
            01_G01_R01_Color_<timestamp-1>.png
            01_G01_R01_Color_<timestamp-2>.png
            ...
        R02/
        R03/
        ...
    G02/
    G03/
    ...
user02/
user03/
...
user12/
README.md
```

- Each raw video in `.bag` format was converted to a sequence of `.png` frames, then converted again to `.mp4` files using `ffmpeg`
- Each video is stored under it's corresponding `userXX/GXX/` directory
- For example: `user01/G01/R01.mp4`
#!/bin/bash

python3 ../inference_weave.py \
--dataset "MAdAiLab/amazon-attrprompt" \
--prompt """
Text: {text}

Given the above amazon review for various categories of product:

0: magazines
1: camera_photo
2: office_products
3: kitchen
4: cell_phones_service
5: computer_video_games
6: grocery_and_gourmet_food
7: tools_hardware
8: automotive
9: music_album
10: health_and_personal_care
11: electronics
12: outdoor_living
13: video
14: apparel
15: toys_games
16: sports_outdoors
17: books
18: software
19: baby
20: musical_and_instruments
21: beauty
22: jewelry_and_watches

Provide your classification in a concise and definitive manner, outputting the corresponding class label (0-22). 
Classification Label: """ \
--name "amazon_zero_shot_test" \
--batch_size 1000
#--test_samples 500
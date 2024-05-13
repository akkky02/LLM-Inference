#!/bin/bash

../inference_weave.py \
--dataset "MAdAiLab/twitter_disaster" \
--prompt "Given the following tweet:

"{text}"

0: negative
1: positive

What is your answer? Please respond with 0 or 1.

Answer: " \
--name "twitter_zero_shot_test"
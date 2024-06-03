#!/bin/bash

python3 ../inference_weave.py \
--dataset "MAdAiLab/lex_glue_scotus" \
--prompt """
Text: {text}

Given the above Supreme Court opinion document:

0: Criminal Procedure
1: Civil Rights
2: First Amendment
3: Due Process
4: Privacy
5: Attorneys
6: Unions
7: Economic Activity
8: Judicial Power
9: Federalism
10: Interstate Relations
11: Federal Taxation
12: Miscellaneous

Provide your classification in a concise and definitive manner, outputting the corresponding class label (0-12).
Classification Label: """ \
--name "scotus_zero_shot" \
--batch_size 1000
#--test_samples 500
#!/bin/bash

python3 ../inference_weave.py \
--dataset "MAdAiLab/lex_glue_ledgar" \
--prompt """
Given the following legal provisions in contracts text:
{text}

0: Adjustments
1: Agreements
2: Amendments
3: Anti-Corruption Laws
4: Applicable Laws
5: Approvals
6: Arbitration
7: Assignments
8: Assigns
9: Authority
10: Authorizations
11: Base Salary
12: Benefits
13: Binding Effects
14: Books
15: Brokers
16: Capitalization
17: Change In Control
18: Closings
19: Compliance With Laws
20: Confidentiality
21: Consent To Jurisdiction
22: Consents
23: Construction
24: Cooperation
25: Costs
26: Counterparts
27: Death
28: Defined Terms
29: Definitions
30: Disability
31: Disclosures
32: Duties
33: Effective Dates
34: Effectiveness
35: Employment
36: Enforceability
37: Enforcements
38: Entire Agreements
39: Erisa
40: Existence
41: Expenses
42: Fees
43: Financial Statements
44: Forfeitures
45: Further Assurances
46: General
47: Governing Laws
48: Headings
49: Indemnifications
50: Indemnity
51: Insurances
52: Integration
53: Intellectual Property
54: Interests
55: Interpretations
56: Jurisdictions
57: Liens
58: Litigations
59: Miscellaneous
60: Modifications
61: No Conflicts
62: No Defaults
63: No Waivers
64: Non-Disparagement
65: Notices
66: Organizations
67: Participations
68: Payments
69: Positions
70: Powers
71: Publicity
72: Qualifications
73: Records
74: Releases
75: Remedies
76: Representations
77: Sales
78: Sanctions
79: Severability
80: Solvency
81: Specific Performance
82: Submission To Jurisdiction
83: Subsidiaries
84: Successors
85: Survival
86: Tax Withholdings
87: Taxes
88: Terminations
89: Terms
90: Titles
91: Transactions With Affiliates
92: Use Of Proceeds
93: Vacations
94: Venues
95: Vesting
96: Waiver Of Jury Trials
97: Waivers
98: Warranties
99: Withholdings

Provide your classification in a concise and definitive manner, outputting the corresponding class label (0-99). 
Do not provide any additional commentary or explanation beyond the classification itself.
Classification Label: """ \
--name "ledgar_zero_shot" \
--batch_size 1000
#--test_samples 500
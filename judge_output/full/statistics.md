
# Judge Results — Statistics Report

from `judge_output/full/judge_results.jsonl`


## 1. Overall

**Total entries:** 3409  |  **All-pass:** 2441/3409 = 71.6%

| Metric | Mean | Pass | Fail |
|---|---|---|---|
| recoverability | 0.8686 | 2961 | 448 |
| naturalness | 0.9349 | 3187 | 222 |
| semantic_preservation | 0.7319 | 2495 | 914 |

## 2. Per Dataset

| Dataset | n | Recoverability | Naturalness | Semantic Preservation | All-Pass |
|---|---|---|---|---|---|
| humaneval | 487 | 0.7967 (99 fail) | 0.9302 (34 fail) | 0.7166 (138 fail) | 304/487 (62.4%) |
| mbpp | 2922 | 0.8806 (349 fail) | 0.9357 (188 fail) | 0.7344 (776 fail) | 2137/2922 (73.1%) |

## 3. Per Mutation Type


### LV — Lexical Variation (n=1137)
| Metric | Mean | Pass | Fail |
|---|---|---|---|
| recoverability | 0.9815 | 1116 | 21 |
| naturalness | 0.9886 | 1124 | 13 |
| semantic_preservation | 0.9226 | 1049 | 88 |
| lexical_compliance | 1.0000 | 1137 | 0 |
| **all-pass** | **0.9226** | 1049 | 88 |

### SF — Style/Formatting (n=1138)
| Metric | Mean | Pass | Fail |
|---|---|---|---|
| recoverability | 0.8937 | 1017 | 121 |
| naturalness | 0.9965 | 1134 | 4 |
| semantic_preservation | 0.8541 | 972 | 166 |
| formatting_compliance | 1.0000 | 1138 | 0 |
| **all-pass** | **0.8251** | 939 | 199 |

### US — Underspecification (n=1134)
| Metric | Mean | Pass | Fail |
|---|---|---|---|
| recoverability | 0.7302 | 828 | 306 |
| naturalness | 0.8192 | 929 | 205 |
| semantic_preservation | 0.4180 | 474 | 660 |
| underspec_compliance | 0.8836 | 1002 | 132 |
| **all-pass** | **0.2875** | 326 | 808 |

## 4. Per Dataset x Mutation Type

| Dataset | Type | n | Recoverability | Naturalness | Semantic Pres. | Type-Specific | All-Pass |
|---|---|---|---|---|---|---|---|
| humaneval | LV | 163 | 0.9939 (1 fail) | 0.9877 (2 fail) | 0.9571 (7 fail) | 1.0000 (0 fail) | 156/163 (95.7%) |
| humaneval | SF | 164 | 0.5122 (80 fail) | 0.9939 (1 fail) | 0.5305 (77 fail) | 1.0000 (0 fail) | 59/164 (36.0%) |
| humaneval | US | 160 | 0.8875 (18 fail) | 0.8063 (31 fail) | 0.6625 (54 fail) | 0.8938 (17 fail) | 74/160 (46.2%) |
| mbpp | LV | 974 | 0.9795 (20 fail) | 0.9887 (11 fail) | 0.9168 (81 fail) | 1.0000 (0 fail) | 893/974 (91.7%) |
| mbpp | SF | 974 | 0.9579 (41 fail) | 0.9969 (3 fail) | 0.9086 (89 fail) | 1.0000 (0 fail) | 880/974 (90.3%) |
| mbpp | US | 974 | 0.7043 (288 fail) | 0.8214 (174 fail) | 0.3778 (606 fail) | 0.8819 (115 fail) | 252/974 (25.9%) |

## 5. Failure Co-occurrence by Mutation Type


### LV — 88 entries with at least one failure
| Failed Metrics Combo | Count |
|---|---|
| ['semantic_preservation'] | 66 |
| ['naturalness', 'recoverability', 'semantic_preservation'] | 12 |
| ['recoverability', 'semantic_preservation'] | 9 |
| ['naturalness', 'semantic_preservation'] | 1 |

### SF — 199 entries with at least one failure
| Failed Metrics Combo | Count |
|---|---|
| ['recoverability', 'semantic_preservation'] | 88 |
| ['semantic_preservation'] | 76 |
| ['recoverability'] | 31 |
| ['naturalness', 'recoverability', 'semantic_preservation'] | 2 |
| ['naturalness'] | 2 |

### US — 808 entries with at least one failure
| Failed Metrics Combo | Count |
|---|---|
| ['semantic_preservation'] | 334 |
| ['naturalness', 'recoverability', 'semantic_preservation'] | 165 |
| ['recoverability', 'semantic_preservation'] | 141 |
| ['underspec_compliance'] | 127 |
| ['naturalness', 'semantic_preservation'] | 19 |
| ['naturalness'] | 17 |
| ['naturalness', 'underspec_compliance'] | 4 |
| ['semantic_preservation', 'underspec_compliance'] | 1 |

## 6. Per-Metric Failure Counts

| Metric | Failures | Applicable n |
|---|---|---|
| recoverability | 448 | 3409 |
| naturalness | 222 | 3409 |
| semantic_preservation | 914 | 3409 |
| lexical_compliance | 0 | 1137 |
| formatting_compliance | 0 | 1138 |
| underspec_compliance | 132 | 1134 |

## 7. Task-Level All-Pass Summary (across all 3 mutation types)

| Dataset | Total Tasks | Fully Passing All Types | Rate | Failing Any Type |
|---|---|---|---|---|
| humaneval | 164 | 26 | 15.8% | 138 |
| mbpp | 974 | 210 | 21.6% | 764 |

*A task is 'fully passing' only if all 3 mutation types (LV, SF, US) pass all their respective metrics.*

## 8. Entries Failing 3+ Metrics


### LV — 12 entries failing 3+ metrics
| Judge ID | Failed Metrics |
|---|---|
| 139__LV | recoverability, naturalness, semantic_preservation |
| 169__LV | recoverability, naturalness, semantic_preservation |
| 234__LV | recoverability, naturalness, semantic_preservation |
| 312__LV | recoverability, naturalness, semantic_preservation |
| 52__LV | recoverability, naturalness, semantic_preservation |
| 583__LV | recoverability, naturalness, semantic_preservation |
| 608__LV | recoverability, naturalness, semantic_preservation |
| 636__LV | recoverability, naturalness, semantic_preservation |
| 714__LV | recoverability, naturalness, semantic_preservation |
| 80__LV | recoverability, naturalness, semantic_preservation |
| 814__LV | recoverability, naturalness, semantic_preservation |
| HumanEval/117__LV | recoverability, naturalness, semantic_preservation |

### SF — 2 entries failing 3+ metrics
| Judge ID | Failed Metrics |
|---|---|
| 363__SF | recoverability, naturalness, semantic_preservation |
| HumanEval/154__SF | recoverability, naturalness, semantic_preservation |

### US — 165 entries failing 3+ metrics
| Judge ID | Failed Metrics |
|---|---|
| 103__US | recoverability, naturalness, semantic_preservation |
| 11__US | recoverability, naturalness, semantic_preservation |
| 122__US | recoverability, naturalness, semantic_preservation |
| 128__US | recoverability, naturalness, semantic_preservation |
| 135__US | recoverability, naturalness, semantic_preservation |
| 140__US | recoverability, naturalness, semantic_preservation |
| 147__US | recoverability, naturalness, semantic_preservation |
| 169__US | recoverability, naturalness, semantic_preservation |
| 172__US | recoverability, naturalness, semantic_preservation |
| 201__US | recoverability, naturalness, semantic_preservation |
| 202__US | recoverability, naturalness, semantic_preservation |
| 204__US | recoverability, naturalness, semantic_preservation |
| 207__US | recoverability, naturalness, semantic_preservation |
| 21__US | recoverability, naturalness, semantic_preservation |
| 221__US | recoverability, naturalness, semantic_preservation |
| 225__US | recoverability, naturalness, semantic_preservation |
| 226__US | recoverability, naturalness, semantic_preservation |
| 227__US | recoverability, naturalness, semantic_preservation |
| 229__US | recoverability, naturalness, semantic_preservation |
| 230__US | recoverability, naturalness, semantic_preservation |
| 234__US | recoverability, naturalness, semantic_preservation |
| 235__US | recoverability, naturalness, semantic_preservation |
| 241__US | recoverability, naturalness, semantic_preservation |
| 244__US | recoverability, naturalness, semantic_preservation |
| 259__US | recoverability, naturalness, semantic_preservation |
| 264__US | recoverability, naturalness, semantic_preservation |
| 273__US | recoverability, naturalness, semantic_preservation |
| 283__US | recoverability, naturalness, semantic_preservation |
| 288__US | recoverability, naturalness, semantic_preservation |
| 289__US | recoverability, naturalness, semantic_preservation |
| 290__US | recoverability, naturalness, semantic_preservation |
| 291__US | recoverability, naturalness, semantic_preservation |
| 294__US | recoverability, naturalness, semantic_preservation |
| 298__US | recoverability, naturalness, semantic_preservation |
| 299__US | recoverability, naturalness, semantic_preservation |
| 29__US | recoverability, naturalness, semantic_preservation |
| 307__US | recoverability, naturalness, semantic_preservation |
| 309__US | recoverability, naturalness, semantic_preservation |
| 311__US | recoverability, naturalness, semantic_preservation |
| 318__US | recoverability, naturalness, semantic_preservation |
| 321__US | recoverability, naturalness, semantic_preservation |
| 322__US | recoverability, naturalness, semantic_preservation |
| 323__US | recoverability, naturalness, semantic_preservation |
| 325__US | recoverability, naturalness, semantic_preservation |
| 32__US | recoverability, naturalness, semantic_preservation |
| 331__US | recoverability, naturalness, semantic_preservation |
| 338__US | recoverability, naturalness, semantic_preservation |
| 345__US | recoverability, naturalness, semantic_preservation |
| 346__US | recoverability, naturalness, semantic_preservation |
| 35__US | recoverability, naturalness, semantic_preservation |
| 360__US | recoverability, naturalness, semantic_preservation |
| 368__US | recoverability, naturalness, semantic_preservation |
| 385__US | recoverability, naturalness, semantic_preservation |
| 389__US | recoverability, naturalness, semantic_preservation |
| 392__US | recoverability, naturalness, semantic_preservation |
| 393__US | recoverability, naturalness, semantic_preservation |
| 394__US | recoverability, naturalness, semantic_preservation |
| 396__US | recoverability, naturalness, semantic_preservation |
| 404__US | recoverability, naturalness, semantic_preservation |
| 410__US | recoverability, naturalness, semantic_preservation |
| 412__US | recoverability, naturalness, semantic_preservation |
| 415__US | recoverability, naturalness, semantic_preservation |
| 418__US | recoverability, naturalness, semantic_preservation |
| 432__US | recoverability, naturalness, semantic_preservation |
| 433__US | recoverability, naturalness, semantic_preservation |
| 436__US | recoverability, naturalness, semantic_preservation |
| 437__US | recoverability, naturalness, semantic_preservation |
| 443__US | recoverability, naturalness, semantic_preservation |
| 444__US | recoverability, naturalness, semantic_preservation |
| 453__US | recoverability, naturalness, semantic_preservation |
| 456__US | recoverability, naturalness, semantic_preservation |
| 457__US | recoverability, naturalness, semantic_preservation |
| 45__US | recoverability, naturalness, semantic_preservation |
| 460__US | recoverability, naturalness, semantic_preservation |
| 463__US | recoverability, naturalness, semantic_preservation |
| 465__US | recoverability, naturalness, semantic_preservation |
| 472__US | recoverability, naturalness, semantic_preservation |
| 476__US | recoverability, naturalness, semantic_preservation |
| 480__US | recoverability, naturalness, semantic_preservation |
| 481__US | recoverability, naturalness, semantic_preservation |
| 490__US | recoverability, naturalness, semantic_preservation |
| 50__US | recoverability, naturalness, semantic_preservation |
| 524__US | recoverability, naturalness, semantic_preservation |
| 528__US | recoverability, naturalness, semantic_preservation |
| 529__US | recoverability, naturalness, semantic_preservation |
| 531__US | recoverability, naturalness, semantic_preservation |
| 536__US | recoverability, naturalness, semantic_preservation |
| 540__US | recoverability, naturalness, semantic_preservation |
| 546__US | recoverability, naturalness, semantic_preservation |
| 570__US | recoverability, naturalness, semantic_preservation |
| 576__US | recoverability, naturalness, semantic_preservation |
| 577__US | recoverability, naturalness, semantic_preservation |
| 57__US | recoverability, naturalness, semantic_preservation |
| 580__US | recoverability, naturalness, semantic_preservation |
| 589__US | recoverability, naturalness, semantic_preservation |
| 591__US | recoverability, naturalness, semantic_preservation |
| 593__US | recoverability, naturalness, semantic_preservation |
| 59__US | recoverability, naturalness, semantic_preservation |
| 5__US | recoverability, naturalness, semantic_preservation |
| 603__US | recoverability, naturalness, semantic_preservation |
| 606__US | recoverability, naturalness, semantic_preservation |
| 614__US | recoverability, naturalness, semantic_preservation |
| 620__US | recoverability, naturalness, semantic_preservation |
| 624__US | recoverability, naturalness, semantic_preservation |
| 625__US | recoverability, naturalness, semantic_preservation |
| 62__US | recoverability, naturalness, semantic_preservation |
| 641__US | recoverability, naturalness, semantic_preservation |
| 657__US | recoverability, naturalness, semantic_preservation |
| 661__US | recoverability, naturalness, semantic_preservation |
| 674__US | recoverability, naturalness, semantic_preservation |
| 67__US | recoverability, naturalness, semantic_preservation |
| 681__US | recoverability, naturalness, semantic_preservation |
| 68__US | recoverability, naturalness, semantic_preservation |
| 707__US | recoverability, naturalness, semantic_preservation |
| 708__US | recoverability, naturalness, semantic_preservation |
| 711__US | recoverability, naturalness, semantic_preservation |
| 716__US | recoverability, naturalness, semantic_preservation |
| 718__US | recoverability, naturalness, semantic_preservation |
| 737__US | recoverability, naturalness, semantic_preservation |
| 749__US | recoverability, naturalness, semantic_preservation |
| 757__US | recoverability, naturalness, semantic_preservation |
| 758__US | recoverability, naturalness, semantic_preservation |
| 765__US | recoverability, naturalness, semantic_preservation |
| 767__US | recoverability, naturalness, semantic_preservation |
| 779__US | recoverability, naturalness, semantic_preservation |
| 77__US | recoverability, naturalness, semantic_preservation |
| 788__US | recoverability, naturalness, semantic_preservation |
| 791__US | recoverability, naturalness, semantic_preservation |
| 805__US | recoverability, naturalness, semantic_preservation |
| 807__US | recoverability, naturalness, semantic_preservation |
| 826__US | recoverability, naturalness, semantic_preservation |
| 828__US | recoverability, naturalness, semantic_preservation |
| 836__US | recoverability, naturalness, semantic_preservation |
| 842__US | recoverability, naturalness, semantic_preservation |
| 849__US | recoverability, naturalness, semantic_preservation |
| 84__US | recoverability, naturalness, semantic_preservation |
| 852__US | recoverability, naturalness, semantic_preservation |
| 863__US | recoverability, naturalness, semantic_preservation |
| 869__US | recoverability, naturalness, semantic_preservation |
| 874__US | recoverability, naturalness, semantic_preservation |
| 893__US | recoverability, naturalness, semantic_preservation |
| 903__US | recoverability, naturalness, semantic_preservation |
| 90__US | recoverability, naturalness, semantic_preservation |
| 912__US | recoverability, naturalness, semantic_preservation |
| 915__US | recoverability, naturalness, semantic_preservation |
| 920__US | recoverability, naturalness, semantic_preservation |
| 922__US | recoverability, naturalness, semantic_preservation |
| 924__US | recoverability, naturalness, semantic_preservation |
| 932__US | recoverability, naturalness, semantic_preservation |
| 934__US | recoverability, naturalness, semantic_preservation |
| 937__US | recoverability, naturalness, semantic_preservation |
| 947__US | recoverability, naturalness, semantic_preservation |
| 950__US | recoverability, naturalness, semantic_preservation |
| 951__US | recoverability, naturalness, semantic_preservation |
| 957__US | recoverability, naturalness, semantic_preservation |
| 965__US | recoverability, naturalness, semantic_preservation |
| 970__US | recoverability, naturalness, semantic_preservation |
| HumanEval/124__US | recoverability, naturalness, semantic_preservation |
| HumanEval/159__US | recoverability, naturalness, semantic_preservation |
| HumanEval/17__US | recoverability, naturalness, semantic_preservation |
| HumanEval/27__US | recoverability, naturalness, semantic_preservation |
| HumanEval/30__US | recoverability, naturalness, semantic_preservation |
| HumanEval/33__US | recoverability, naturalness, semantic_preservation |
| HumanEval/81__US | recoverability, naturalness, semantic_preservation |
| HumanEval/89__US | recoverability, naturalness, semantic_preservation |

## 9. All Failing Entries


### LV — 88 failing entries
| Judge ID | Recoverability | Naturalness | Semantic Pres. | Lexical Compliance | Failed Metrics |
|---|---|---|---|---|---|
| 112__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 114__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 116__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 120__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 136__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 139__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 143__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 150__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 151__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 163__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 169__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 191__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 196__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 202__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 215__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 216__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 234__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 235__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 259__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 261__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 268__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 28__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 307__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 312__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 335__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 343__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 348__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 349__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 354__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 355__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 386__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 399__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 432__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 440__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 484__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 498__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 510__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 526__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 52__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 539__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 551__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 553__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 574__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 583__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 593__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 608__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 636__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 645__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 651__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 662__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 693__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 703__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 706__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 714__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 718__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 722__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 737__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 742__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 746__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 750__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 762__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 766__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 780__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 784__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 789__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 791__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 792__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 80__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 814__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 82__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 859__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 869__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 871__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 884__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 885__LV | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 918__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 921__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 927__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 932__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 949__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| 94__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/116__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/117__LV | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/135__LV | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/143__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/35__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/53__LV | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/92__LV | 1 | 1 | 0 | 1 | semantic_preservation |

### SF — 199 failing entries
| Judge ID | Recoverability | Naturalness | Semantic Pres. | Formatting Compliance | Failed Metrics |
|---|---|---|---|---|---|
| 102__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 117__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 128__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 139__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 142__SF | 0 | 1 | 1 | 1 | recoverability |
| 146__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 153__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 165__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 16__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 172__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 178__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 186__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 196__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 198__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 1__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 200__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 209__SF | 1 | 0 | 1 | 1 | naturalness |
| 210__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 232__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 239__SF | 0 | 1 | 1 | 1 | recoverability |
| 245__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 272__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 274__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 285__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 300__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 307__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 328__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 363__SF | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 369__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 385__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 389__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 392__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 398__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 400__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 401__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 424__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 428__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 432__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 438__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 444__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 445__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 447__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 452__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 456__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 464__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 503__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 506__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 512__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 516__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 548__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 574__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 583__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 591__SF | 0 | 1 | 1 | 1 | recoverability |
| 616__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 617__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 631__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 639__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 63__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 648__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 652__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 653__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 65__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 674__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 682__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 687__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 69__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 710__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 71__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 729__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 736__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 737__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 738__SF | 1 | 0 | 1 | 1 | naturalness |
| 755__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 783__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 784__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 795__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 796__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 812__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 81__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 841__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 84__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 859__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 865__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 881__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 888__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 90__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 910__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 912__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 921__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 935__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 938__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 950__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 965__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| 99__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/0__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/100__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/101__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/102__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/103__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/105__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/108__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/109__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/10__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/111__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/114__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/115__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/116__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/117__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/118__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/119__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/11__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/120__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/121__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/122__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/125__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/126__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/127__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/129__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/12__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/130__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/131__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/132__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/133__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/135__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/137__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/138__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/139__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/140__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/141__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/144__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/146__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/147__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/149__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/14__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/150__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/153__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/154__SF | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/156__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/158__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/15__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/162__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/163__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/17__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/19__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/1__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/21__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/22__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/23__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/24__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/25__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/26__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/27__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/28__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/2__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/31__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/32__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/34__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/35__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/36__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/37__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/38__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/39__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/3__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/40__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/41__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/43__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/44__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/47__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/48__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/49__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/4__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/50__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/53__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/56__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/57__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/59__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/5__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/63__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/65__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/68__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/6__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/70__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/71__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/72__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/73__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/75__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/76__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/78__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/79__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/80__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/85__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/87__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/8__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/91__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/92__SF | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/95__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/98__SF | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/99__SF | 0 | 1 | 1 | 1 | recoverability |
| HumanEval/9__SF | 0 | 1 | 1 | 1 | recoverability |

### US — 808 failing entries
| Judge ID | Recoverability | Naturalness | Semantic Pres. | Underspec Compliance | Failed Metrics |
|---|---|---|---|---|---|
| 100__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 101__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 103__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 105__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 107__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 108__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 109__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 110__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 111__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 112__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 114__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 117__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 119__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 11__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 120__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 122__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 123__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 125__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 126__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 128__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 130__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 132__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 134__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 135__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 13__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 140__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 142__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 143__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 145__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 146__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 147__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 148__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 149__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 152__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 153__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 154__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 155__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 156__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 157__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 15__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 160__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 162__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 164__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 166__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 168__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 169__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 16__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 170__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 172__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 177__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 178__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 179__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 17__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 181__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 182__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 183__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 186__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 187__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 188__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 189__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 18__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 190__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 192__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 193__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 194__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 195__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 197__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 198__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 19__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 1__US | 1 | 0 | 1 | 0 | naturalness, underspec_compliance |
| 200__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 201__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 202__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 204__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 206__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 207__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 208__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 209__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 20__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 210__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 211__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 212__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 215__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 216__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 217__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 218__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 219__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 21__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 220__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 221__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 222__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 223__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 225__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 226__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 227__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 229__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 22__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 230__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 231__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 233__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 234__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 235__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 236__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 237__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 238__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 239__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 23__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 241__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 243__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 244__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 245__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 246__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 247__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 248__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 24__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 250__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 251__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 253__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 254__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 255__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 259__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 25__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 260__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 262__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 264__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 265__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 268__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 269__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 26__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 270__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 271__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 272__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 273__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 274__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 275__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 277__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 278__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 279__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 280__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 282__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 283__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 285__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 286__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 288__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 289__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 290__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 291__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 292__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 294__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 295__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 297__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 298__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 299__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 29__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 2__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 300__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 301__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 302__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 304__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 305__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 306__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 307__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 308__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 309__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 30__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 310__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 311__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 313__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 314__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 315__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 316__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 317__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 318__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 31__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 321__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 322__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 323__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 324__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 325__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 326__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 327__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 328__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 329__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 32__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 331__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 333__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 334__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 337__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 338__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 339__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 33__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 340__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 341__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 342__US | 1 | 0 | 1 | 0 | naturalness, underspec_compliance |
| 343__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 344__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 345__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 346__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 348__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 34__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 350__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 351__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 354__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 355__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 35__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 360__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 361__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 362__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 363__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 364__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 365__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 366__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 367__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 368__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 369__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 36__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 370__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 371__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 376__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 378__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 37__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 381__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 383__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 384__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 385__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 386__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 387__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 388__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 389__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 38__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 392__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 393__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 394__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 395__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 396__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 397__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 398__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 399__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 39__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 3__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 400__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 401__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 403__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 404__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 405__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 406__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 408__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 409__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 410__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 412__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 413__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 415__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 416__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 417__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 418__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 419__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 421__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 424__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 426__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 428__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 42__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 431__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 432__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 433__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 434__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 435__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 436__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 437__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 438__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 439__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 43__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 440__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 442__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 443__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 444__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 449__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 44__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 450__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 452__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 453__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 454__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 456__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 457__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 45__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 460__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 461__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 463__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 464__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 465__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 466__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 468__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 469__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 470__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 471__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 472__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 474__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 475__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 476__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 478__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 479__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 47__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 480__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 481__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 483__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 485__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 488__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 489__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 48__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 490__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 492__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 493__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 495__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 496__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 497__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 498__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 499__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 49__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 4__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 500__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 502__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 503__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 505__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 506__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 508__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 509__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 50__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 511__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 512__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 513__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 515__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 516__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 517__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 518__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 519__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 51__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 521__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 522__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 523__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 524__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 528__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 529__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 52__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 530__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 531__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 533__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 535__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 536__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 537__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 540__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 541__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 544__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 545__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 546__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 547__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 548__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 549__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 54__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 551__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 552__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 554__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 556__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 557__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 559__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 55__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 562__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 564__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 565__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 566__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 567__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 568__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 569__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 56__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 570__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 571__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 572__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 573__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 574__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 575__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 576__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 577__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 578__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 57__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 580__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 581__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 583__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 585__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 586__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 587__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 588__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 589__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 58__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 591__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 593__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 594__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 595__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 596__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 597__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 598__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 59__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 5__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 600__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 601__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 602__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 603__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 604__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 606__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 608__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 609__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 60__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 610__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 611__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 612__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 613__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 614__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 615__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 616__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 61__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 620__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 621__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 622__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 624__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 625__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 626__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 627__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 629__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 62__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 630__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 632__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 633__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 634__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 635__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 637__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 639__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 63__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 641__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 642__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 643__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 645__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 646__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 647__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 648__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 64__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 650__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 652__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 655__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 656__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 657__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 658__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 65__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 660__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 661__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 662__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 663__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 664__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 665__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 666__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 667__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 668__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 66__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 671__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 672__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 673__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 674__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 675__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 676__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 677__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 679__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 67__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 680__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 681__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 684__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 686__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 687__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 688__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 68__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 690__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 692__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 694__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 695__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 696__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 698__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 699__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 69__US | 1 | 0 | 1 | 1 | naturalness |
| 6__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 701__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 703__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 704__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 705__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 707__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 708__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 710__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 711__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 712__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 714__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 715__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 716__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 717__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 718__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 719__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 720__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 721__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 722__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 724__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 726__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 727__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 728__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 729__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 731__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 733__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 734__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 735__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 736__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 737__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 738__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 739__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 740__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 742__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 743__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 744__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 745__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 746__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 749__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 74__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 751__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 752__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 753__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 754__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 755__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 756__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 757__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 758__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 759__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 75__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 760__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 761__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 763__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 764__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 765__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 766__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 767__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 768__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 771__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 772__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 773__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 774__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 776__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 777__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 778__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 779__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 77__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 780__US | 1 | 0 | 1 | 1 | naturalness |
| 781__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 782__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 783__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 784__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 785__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 786__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 787__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 788__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 789__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 790__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 791__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 793__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 794__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 795__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 796__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 797__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 798__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 799__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 79__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 7__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 801__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 802__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 805__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 806__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 807__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 809__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 80__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 810__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 812__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 813__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 815__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 818__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 819__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 822__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 823__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 824__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 825__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 826__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 828__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 829__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 831__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 832__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 833__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 834__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 836__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 837__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 839__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 83__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 841__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 842__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 843__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 844__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 845__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 846__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 847__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 849__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 84__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 851__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 852__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 853__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 855__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 856__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 857__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 858__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 860__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 862__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 863__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 865__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 867__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 868__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 869__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 86__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 870__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 871__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 872__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 873__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 874__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 875__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 876__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 877__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 878__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 879__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 87__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 880__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 881__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 885__US | 1 | 1 | 0 | 0 | semantic_preservation, underspec_compliance |
| 886__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 88__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 890__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 891__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 892__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| 893__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 894__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 895__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 897__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 898__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 899__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 89__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 8__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 901__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 902__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 903__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 908__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 909__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 90__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 910__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 911__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 912__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 913__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 915__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 916__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 917__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 919__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 920__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 921__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 922__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 923__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 924__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 925__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 926__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 927__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 928__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 929__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 92__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 930__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 932__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 934__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 935__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 936__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 937__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 938__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 939__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 93__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 940__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 941__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 943__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 944__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 945__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 947__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 948__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 949__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 94__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 950__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 951__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 952__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 953__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 955__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 956__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 957__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 958__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 959__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 95__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 960__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 961__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 962__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 963__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 964__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 965__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 966__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 967__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 968__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 969__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 96__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 970__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| 971__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 972__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 973__US | 1 | 1 | 0 | 1 | semantic_preservation |
| 974__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 97__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| 98__US | 1 | 1 | 1 | 0 | underspec_compliance |
| 99__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/0__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/100__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/102__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/103__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/105__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/106__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/107__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/108__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/109__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/112__US | 1 | 0 | 1 | 0 | naturalness, underspec_compliance |
| HumanEval/116__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/117__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/120__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/122__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/124__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/125__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/127__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/129__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/12__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/133__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/134__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/135__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/136__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/137__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/138__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/139__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/13__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/141__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/142__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/146__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/147__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/149__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/150__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/151__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/152__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/153__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/154__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/156__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/158__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/159__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/160__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/161__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/17__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/18__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/1__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/24__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/26__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/27__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/29__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/30__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/32__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/33__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/36__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/37__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/38__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/39__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/42__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/49__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/51__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/54__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/57__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/58__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/5__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/60__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/64__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/65__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/67__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/68__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/6__US | 1 | 1 | 1 | 0 | underspec_compliance |
| HumanEval/71__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/74__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/75__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/76__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/79__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/80__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/81__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/86__US | 1 | 0 | 0 | 1 | naturalness, semantic_preservation |
| HumanEval/87__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/89__US | 0 | 0 | 0 | 1 | recoverability, naturalness, semantic_preservation |
| HumanEval/8__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/90__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/92__US | 1 | 1 | 0 | 1 | semantic_preservation |
| HumanEval/93__US | 1 | 0 | 1 | 0 | naturalness, underspec_compliance |
| HumanEval/94__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
| HumanEval/97__US | 1 | 0 | 1 | 1 | naturalness |
| HumanEval/98__US | 0 | 1 | 0 | 1 | recoverability, semantic_preservation |
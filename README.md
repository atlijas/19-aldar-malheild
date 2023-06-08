# 19. aldar málheild

(English below)

### 🇮🇸 Íslenska 🇮🇸
* Sækið ljóslestrarvilluleiðréttingarlíkanið:  
`curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/309{/ocr-p-p.zip}`

* Afþjappið (eða hvernig sem er en data/- og models/-möppurnar þurfa að vera í möppu sem heitir frsq/):  
`unzip ocr-p-p.zip "frsq/**/*"`

* Sækið `mideind/IceBERT` í `transformer_models/`:  
`git clone https://huggingface.co/mideind/IceBERT transformer_models/IceBERT`

* Sækið `mideind/yfirlestur-icelandic-correction-byt5` í `transformer_models/`:  
`git clone https://huggingface.co/mideind/yfirlestur-icelandic-correction-byt5 transformer_models/yfirlestur-icelandic-correction-byt5`

* Hægt er að nota skriftuna `single_main.py` til þess að leiðrétta ljóslestrarvillur í einni skrá, nútímavæða stafsetningu og sameina í henni línur í málsgreinar:  
`python3 single_main.py --file test_data/original/short_test.txt`


* Hægt er að nota flaggið `--transform-only` þess að leiðrétta einungis ljóslestrarvillur:  
`python3 single_main.py --file test_data/original/short_test.txt --transform-only`

Skriftan `main.py` krefst þess að í `all_txt/` séu ljóslesnar skrár og að möppustrúktúrinn sé með þessum hætti: `all_txt/tímarit/tölublað/bls.txt`. Í möppunni `all_txt/tímarit/tölublað` þarf einnig að vera skrá sem heitir `.issue.json`. Því hentar `single_main.py` sennilega betur fyrir almenna notkun þessa tóls.


Kvistur (`utils/kvistur/`) er hugbúnaður [Jóns Friðriks Daðasonar](https://github.com/jonfd/kvistur) og er dreift með [CC BY 4.0-leyfi](https://creativecommons.org/licenses/by/4.0/).

Gögnin úr BÍN (`utils/data/bin_tree.pickle`) eru [Sigrúnarsnið](https://bin.arnastofnun.is/gogn/SH-snid) og er dreift með [CC BY-SA 4.0-leyfi](https://creativecommons.org/licenses/by-sa/4.0/).

Gögnin í `eval_data` eru úr [fyrsta tölublaði Gefnar frá 1870](https://timarit.is/page/2043251#page/n0/mode/2up).

Gögnin í `test_data` eru úr ýmsum tímaritum og dagblöðum sem voru fengin af [timarit.is](https://timarit.is/).

---

### 🇬🇧 English 🇬🇧
* Download the OCR post-processing model:  
`curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/309{/ocr-p-p.zip}`

* Unzip it (or however you'd like, but the data/ and models/ directories need to be in a directory called frsq/):):  
`unzip ocr-p-p.zip "frsq/**/*"`

* Download `mideind/IceBERT` to `transformer_models/`:  
`git clone https://huggingface.co/mideind/IceBERT transformer_models/IceBERT`

* Download `mideind/yfirlestur-icelandic-correction-byt5` to `transformer_models/`:
`git clone https://huggingface.co/mideind/yfirlestur-icelandic-correction-byt5 transformer_models/yfirlestur-icelandic-correction-byt5`

* You can use `single_main.py` to correct OCR errors in a given file, modernize its spelling and merge its lines into sentences.  
`python3 single_main.py --file test_data/original/short_test.txt`

* Use the `--transform-only` flag if you only want to correct OCR errors.  
`python3 single_main.py --file test_data/original/short_test.txt --transform-only`

The `main.py` script assumes that in `all_txt/` there are OCRed files and that the directory structure is as follows: `all_txt/magazine/volume/page.txt`. In the directory `all_txt/magazine/volume` there also needs to be a file called `.issue.json`. Therefore, `single_main.py` is probably better suited for general use of this tool.

Kvistur (`utils/kvistur/`) was made by [Jón Friðrik Daðason](https://github.com/jonfd/kvistur) and is distributed with a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The BÍN data (`utils/data/bin_tree.pickle`) are [Sigrúnarsnið](https://bin.arnastofnun.is/gogn/SH-snid) and are distributed with a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

The data in `eval_data` are from [the first volume of Gefn, 1870](https://timarit.is/page/2043251#page/n0/mode/2up).

The data in `test_data` are from various magazines and newspapers, obtained from [timarit.is](https://timarit.is/).
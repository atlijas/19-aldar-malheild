# 19. aldar málheild

(English below)

### 🇮🇸 Íslenska 🇮🇸
* Sækið ljóslestrarvilluleiðréttingarlíkanið:  
`curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/309{/ocr-p-p.zip}`

* Afþjappið:  
`unzip ocr-p-p.zip "frsq/**/*"`

* Hægt er að nota skriftuna `single_main.py` til þess að leiðrétta ljóslestrarvillur í einni skrá, nútímavæða stafsetningu og sameina í henni línur í málsgreinar:  
`python3 single_main.py --file test_data/original/short_test.txt`


* Hægt er að nota flaggið `--transform-only` þess að leiðrétta einungis ljóslestrarvillur:  
`python3 single_main.py --file test_data/original/short_test.txt --transform-only`

Skriftan `main.py` krefst þess að í `all_txt/` séu ljóslesnar skrár og að möppustrúktúrinn sé með þessum hætti: `all_txt/tímarit/tölublað/bls.txt`. Í möppunni `all_txt/tímarit/tölublað` þarf einnig að vera skrá sem heitir `.issue.json`. Því hentar `single_main.py` sennilega betur fyrir almenna notkun þessa tóls.


Kvistur (`utils/kvistur/`) er hugbúnaður [Jóns Friðriks Daðasonar](https://github.com/jonfd/kvistur) og er dreift með [CC BY 4.0-leyfi](https://creativecommons.org/licenses/by/4.0/).

Gögnin úr BÍN (`utils/data/bin_tree.pickle`) eru [Sigrúnarsnið](https://bin.arnastofnun.is/gogn/SH-snid) og er dreift með [CC BY-SA 4.0-leyfi](https://creativecommons.org/licenses/by-sa/4.0/).


---

### 🇬🇧 English 🇬🇧
* Download the OCR post-processing model:  
`curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/309{/ocr-p-p.zip}`

* Unzip it:  
`unzip ocr-p-p.zip "frsq/**/*"`

* You can use `single_main.py` to correct OCR errors in a given file, modernize its spelling and merge its lines into sentences.  
`python3 single_main.py --file test_data/original/short_test.txt`

* Use the `--transform-only` flag if you only want to correct OCR errors.  
`python3 single_main.py --file test_data/original/short_test.txt --transform-only`

The `main.py` script assumes that in `all_txt/` there are OCRed files and that the directory structure is as follows: `all_txt/magazine/volume/page.txt`. In the directory `all_txt/magazine/volume` there also needs to be a file called `.issue.json`. Therefore, `single_main.py` is probably better suited for general use of this tool.

Kvistur (`utils/kvistur/`) was made by [Jón Friðrik Daðason](https://github.com/jonfd/kvistur) and is distributed with a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The BÍN data (`utils/data/bin_tree.pickle`) are [Sigrúnarsnið](https://bin.arnastofnun.is/gogn/SH-snid) and are distributed with a [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
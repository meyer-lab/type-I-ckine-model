SHELL := /bin/bash

flist = C1 C2 C3 C4 C5 C6

notebooks := $(wildcard *.ipynb)

.PHONY: clean test all testprofile testcover spell

all: pylint.log $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

%.pdf: %.ipynb venv
	. venv/bin/activate && jupyter nbconvert --execute --ExecutePreprocessor.timeout=6000 --to pdf $< --output $@

output/figure%.svg: venv genFigures.py ckine/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg pylint.log
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
	rm -f profile.p* stats.dat .coverage nosetests.xml coverage.xml testResults.xml
	rm -rf html doxy.log graph_all.svg venv ./ckine/data/flow
	find -iname "*.pyc" -delete

spell: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic


download:
	mkdir -p ./ckine/data/flow
	wget -nv -P ./ckine/data/flow/ "https://syno.seas.ucla.edu:9001/gc-cytokines/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip"
	wget -nv -P ./ckine/data/flow/ "https://syno.seas.ucla.edu:9001/gc-cytokines/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT.zip"
	wget -nv -P ./ckine/data/flow/ "https://syno.seas.ucla.edu:9001/gc-cytokines/2019-04-23 Receptor Quant - Beads.zip"
	wget -nv -P ./ckine/data/flow/ "https://syno.seas.ucla.edu:9001/gc-cytokines/4-23_4-26_Receptor quant.zip"
	wget -nv -P ./ckine/data/flow/ "https://syno.seas.ucla.edu:9001/gc-cytokines/2019-05-16 IL7R Quant - Lymphocyte Gated.zip"
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-04-23 Receptor Quant - Beads.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/4-23_4-26_Receptor quant.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-05-16 IL7R Quant - Lymphocyte Gated.zip'

test: venv
	. venv/bin/activate && pytest

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")

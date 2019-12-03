SHELL := /bin/bash

flist = B1 B2 B3 B4 B5 B6 B7 C1 C2 C3

.PHONY: clean test all testprofile testcover spell

all: ckine/ckine.so output/manuscript.html output/manuscript.pdf pylint.log

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv --system-site-packages venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py ckine/ckine.so ckine/figures/figure%.py
	mkdir -p ./Manuscript/Figures
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=./manuscript/ --output-directory=./output --log-level=INFO

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--from=markdown --to=html5 --filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
		--bibliography=output/references.json \
		--csl=common/templates/manubot/style.csl \
		--metadata link-citations=true \
		--include-after-body=common/templates/manubot/default.html \
		--include-after-body=common/templates/manubot/plugins/table-scroll.html \
		--include-after-body=common/templates/manubot/plugins/anchors.html \
		--include-after-body=common/templates/manubot/plugins/accordion.html \
		--include-after-body=common/templates/manubot/plugins/tooltips.html \
		--include-after-body=common/templates/manubot/plugins/jump-to-first.html \
		--include-after-body=common/templates/manubot/plugins/link-highlight.html \
		--include-after-body=common/templates/manubot/plugins/table-of-contents.html \
		--include-after-body=common/templates/manubot/plugins/lightbox.html \
		--mathjax \
		--variable math="" \
		--include-after-body=common/templates/manubot/plugins/math.html \
		--include-after-body=common/templates/manubot/plugins/hypothesis.html \
		--output=output/manuscript.html output/manuscript.md

output/manuscript.pdf: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --from=markdown --to=html5 \
    	--pdf-engine=weasyprint --pdf-engine-opt=--presentational-hints \
    	--filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
    	--bibliography=output/references.json \
    	--csl=common/templates/manubot/style.csl \
    	--metadata link-citations=true \
    	--webtex=https://latex.codecogs.com/svg.latex? \
    	--include-after-body=common/templates/manubot/default.html \
    	--output=output/manuscript.pdf output/manuscript.md

ckine/ckine.so: gcSolver/model.cpp gcSolver/model.hpp gcSolver/reaction.hpp gcSolver/makefile
	cd ./gcSolver && make ckine.so
	cp ./gcSolver/ckine.so ./ckine/ckine.so

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg pylint.log
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true

	rm -f ckine/ckine.so profile.p* stats.dat .coverage nosetests.xml coverage.xml testResults.xml
	rm -rf html doxy.log graph_all.svg valgrind.xml venv ./ckine/data/flow
	find -iname "*.pyc" -delete

spell: manuscript/*.md
	pandoc --lua-filter common/templates/spell.lua manuscript/*.md | sort | uniq -ic

download: venv
	mkdir -p ./ckine/data/flow
	. venv/bin/activate && synapse -u aarmey -p $(SYNAPSE_APIKEY) get syn20506190 --downloadLocation ./ckine/data/flow/
	. venv/bin/activate && synapse -u aarmey -p $(SYNAPSE_APIKEY) get syn20506252 --downloadLocation ./ckine/data/flow/
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT.zip'

test: venv ckine/ckine.so
	. venv/bin/activate && pytest

testcover: venv ckine/ckine.so
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")

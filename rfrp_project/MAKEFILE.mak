#makefile
.PHONY: setup encode run dashboard test package

setup:
	conda env create -f environment.yml

encode:
	python face_encoding.py

run:
	python realtime_recognition.py

dashboard:
	streamlit run dashboard.py

test:
	pytest

package:
	zip -r rfrp_package.zip . -x "*.git*" "*__pycache__*"
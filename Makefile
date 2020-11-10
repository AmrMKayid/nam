PYTHON_MODULE_PATH=name

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete
	find . -name ".ipynb_checkpoints" -type d -delete

format:
	yapf --verbose --in-place --recursive ${PYTHON_MODULE_PATH} --style='{based_on_style: google, indent_width:2, column_limit:80}'
	isort --verbose --force-single-line-imports -y
	docformatter --in-place --recursive ${PYTHON_MODULE_PATH}

SHELL := /bin/bash
ST_DIR := $(HOME)/shimming-toolbox
PYTHON_DIR := python
CLEAN := false

.SILENT: install

# Put it first so that "make" without argument is like "make help".
help:
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

install: ## Run 'make install' to install the plugin [Use flag CLEAN=true for clean install]
	if [[ $(CLEAN) == true || ! -d $(ST_DIR) ]]; then \
		echo "Clean install, deleting $(ST_DIR)."; \
		bash installer/create_st_dir.sh; \
		bash installer/install_conda.sh; \
	elif [[ ! -f $(ST_DIR)/$(PYTHON_DIR)/etc/profile.d/conda.sh ]]; then \
		echo "Conda install not found in $(ST_DIR), installing conda"; \
		bash installer/install_conda.sh; \
	else \
		echo "$(ST_DIR) and conda install found, skipping install of conda"; \
        fi

	bash installer/create_venv.sh
	bash installer/install_plugin.sh
	bash installer/install_shimming_toolbox.sh

run: ## To open FSLeyes with the plugin, run 'make run'
	bash shimming-toolbox.sh

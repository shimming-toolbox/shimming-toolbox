SHELL := /bin/bash
ST_DIR := $(HOME)/shimming-toolbox
PYTHON_DIR := python
CLEAN := true
PLUGIN := true

.SILENT: install

# Put it first so that "make" without argument is like "make help".
help:
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

install: ## Run 'make install' to install Shimming Toolbox [Use flag CLEAN=true for clean install]
	echo "-- Installing Shimming Toolbox -- "
	echo "See our website for more information: https://shimming-toolbox.org"
	echo " "
	if [[ $(CLEAN) == true || ! -d $(ST_DIR) ]]; then \
		echo "Clean install, deleting $(ST_DIR). Warning: If you have previously installed fsleyes-plugin-shimming-toolbox, this will delete it."; \
		bash installer/create_st_dir.sh; \
		bash installer/install_conda.sh; \
	elif [[ ! -f $(ST_DIR)/$(PYTHON_DIR)/etc/profile.d/conda.sh ]]; then \
		echo "Conda install not found in $(ST_DIR), installing conda"; \
		bash installer/install_conda.sh; \
	else \
		echo "$(ST_DIR) and conda install found, skipping install"; \
    fi; \
    bash installer/create_venv.sh; \
    if [[ $(PLUGIN) == false ]]; then \
        echo "Installing Shimming Toolbox"; \
	    bash installer/install_shimming_toolbox.sh; \
	else \
	  	echo "Installing Shimming Toolbox Plugin"; \
		bash installer/install_plugin.sh; \
		echo "Installing Shimming Toolbox"; \
		bash installer/install_shimming_toolbox.sh; \
	fi
	echo "See our website for complete documentation: https://shimming-toolbox.org"

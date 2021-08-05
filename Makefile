SHELL := /bin/bash

# Put it first so that "make" without argument is like "make help".
help:
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

install: ## Run 'make install' to install the plugin
	bash installer/install_conda.sh
	bash installer/install_shimming_toolbox.sh

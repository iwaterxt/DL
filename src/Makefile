# This is the top-level Makefile for Water.
# Also see Water.mk which supplies options and some rules
# used by the Makefiles in the subdirectories.

SHELL := /bin/bash
SUBDIRS = base matrix cumatrix nnet bin

define make_subdir
	@for subdir in $(SUBDIRS) ; do \
	(cd $$subdir && make $1)\
	done;
endef

all:
	$(call make_subdir, all)

install:
	$(call make_subdir, install)
debug:
	$(call make_subdir, debug)
clean:
	$(call make_subdir, clean)

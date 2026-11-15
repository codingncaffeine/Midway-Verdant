# Built with nxdk (https://github.com/XboxDev/nxdk): make NXDK_DIR=/path/to/nxdk
XBE_TITLE = mv\ test\ -\ cpu\ identity
GEN_XISO = mvtest-cpu-identity.iso
SRCS = $(CURDIR)/main.c
NXDK_DIR ?= $(error set NXDK_DIR to an nxdk checkout)

include $(NXDK_DIR)/Makefile

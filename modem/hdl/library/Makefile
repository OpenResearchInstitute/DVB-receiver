
MODULES := 	multiply_accumulate/test \
            fir_filter/test \
            polyphase_filter/test \
            lookup_table/test \
            apsk_modulator/test

.PHONY: $(MODULES)

.PHONY: all
all: $(MODULES)

$(MODULES):
	@cd $@ && $(MAKE)

.PHONY: clean
clean:
	$(foreach TEST, $(MODULES), $(MAKE) -C $(TEST) clean;)

regression:
	$(foreach TEST, $(MODULES), $(MAKE) -C $(TEST) regression;)
PYTHON ?= python

.PHONY: verify verify-declared
verify:
	$(PYTHON) scripts/verify_motion.py --require-complete

verify-declared:
	$(PYTHON) scripts/verify_motion.py

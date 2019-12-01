import cocotb


def print_elements(dut):
	"""
		Traverses the dut printing out information about every element in
		the design
	"""

	# loop through the top level elements
	for design_element in dut:

		print("-"*100)
		print("Found %s : python type = %s: " % (design_element, type(design_element)))
		print("         : _name = %s: ", design_element._name)
		print("         : _path = %s: ", design_element._path)

		# found a sub element to push into
		if type(design_element) == cocotb.handle.HierarchyArrayObject or type(design_element) == cocotb.handle.HierarchyObject:
			print("Pushing into design element")
			print_elements(design_element)


def GSR_control(dut, value):
	"""
		Allows the setting of the GSR (Global Set/Reset) in Xilinx unisim 
		elements.  Will traverse the whole design looking for a GSR signal name
		and will set it.
	"""

	# loop through the top level elements
	for design_element in dut:

		# check that the GSR signal has been found
		if design_element._name == 'GSR':
			design_element.value = value

		# found a sub element to push into
		if type(design_element) == cocotb.handle.HierarchyArrayObject or type(design_element) == cocotb.handle.HierarchyObject:
			GSR_control(design_element, value)


def convert_to_signed(signal, number_bits):
	"""
		Convert the list of input numbers to python signed integer format for
		a given number of bits used to represent the number.
	"""

	output_signal = []

	for sample in signal:
		if sample > (2**(number_bits-1)-1):
			output_signal.append( sample - 2**number_bits )
		else:
			output_signal.append( sample )

	return output_signal


@cocotb.coroutine
def clk_wait(clk_rising, wait_period):
	"""
		Wait for a given number of clock cycles to be more compact for long
		clock delays
	"""

	for i in range(int(wait_period)):
		yield clk_rising


def signed_to_fixedpoint(numbers, number_width, normalised=False):
	"""
		Convert signed numbers to fixed point two's complement version
	"""

	# normalisation factor calculation
	if normalised:
		norm_factor = 2**number_width-1
	else:
		norm_factor = 1

	# support both single numbers and lists
	if type(numbers) == list:
		output = []
		for i, number in enumerate(numbers):
				if number < 0:
					output.append( (2**number_width + number)*norm_factor )
				else:
					output.append( number )
		return output

	else:
		if numbers < 0:
			output = (2**number_width + numbers)*norm_factor
		else:
			output = numbers
		return output


def fixedpoint_to_signed(numbers, number_width, normalised=False):
	"""
		Convert fixed point two's complement number to signed number
	"""

	# normalisation factor calculation
	if normalised:
		norm_factor = 2**number_width-1
	else:
		norm_factor = 1

	# support both single numbers and lists
	if type(numbers) == list:
		output = []
		for i, number in enumerate(numbers):
				if number > 2**(number_width-1)-1:
					output.append( (number - 2**number_width)/norm_factor )
				else:
					output.append( number )
		return output

	else:
		if numbers > 2**(number_width/2)-1:
			output = (numbers - 2**number_width)/norm_factor
		else:
			output = numbers
		return output
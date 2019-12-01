''' Copyright (c) 2014 Potential Ventures Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Potential Ventures Ltd,
      SolarFlare Communications Inc nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL POTENTIAL VENTURES LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. '''
"""
Drivers for Advanced Microcontroller Bus Architecture
"""
import cocotb
from cocotb.triggers import RisingEdge, ReadOnly, Lock
from cocotb.drivers import BusDriver
from cocotb.result import ReturnValue
from cocotb.binary import BinaryValue

import binascii
import array


class AXIProtocolError(Exception):
    pass


class AXI4LiteMaster(BusDriver):
    """
    AXI4-Lite Master

    TODO: Kill all pending transactions if reset is asserted...
    """
    # _signals = ["AWVALID", "AWADDR", "AWREADY",        # Write address channel
    #             "WVALID", "WREADY", "WDATA", "WSTRB",  # Write data channel
    #             "BVALID", "BREADY", "BRESP",           # Write response channel
    #             "ARVALID", "ARADDR", "ARREADY",        # Read address channel
    #             "RVALID", "RREADY", "RRESP", "RDATA"]  # Read data channel

    _signals = ["awvalid", "awaddr", "awready",        # Write address channel
                "wvalid", "wready", "wdata", "wstrb",  # Write data channel
                "bvalid", "bready", "bresp",           # Write response channel
                "arvalid", "araddr", "arready",        # Read address channel
                "rvalid", "rready", "rresp", "rdata"]  # Read data channel

    def __init__(self, entity, name, clock):
        BusDriver.__init__(self, entity, name, clock)

        # Drive some sensible defaults (setimmediatevalue to avoid x asserts)
        # self.bus.AWVALID.setimmediatevalue(0)
        # self.bus.WVALID.setimmediatevalue(0)
        # self.bus.ARVALID.setimmediatevalue(0)
        # self.bus.BREADY.setimmediatevalue(1)
        # self.bus.RREADY.setimmediatevalue(1)
        self.bus.awvalid.setimmediatevalue(0)
        self.bus.wvalid.setimmediatevalue(0)
        self.bus.arvalid.setimmediatevalue(0)
        self.bus.bready.setimmediatevalue(1)
        self.bus.rready.setimmediatevalue(1)

        # Mutex for each channel that we master to prevent contention
        self.write_address_busy = Lock("%s_wabusy" % name)
        self.read_address_busy = Lock("%s_rabusy" % name)
        self.write_data_busy = Lock("%s_wbusy" % name)

    @cocotb.coroutine
    def _send_write_address(self, address, delay=0):
        """
        Send the write address, with optional delay (in clocks)
        """
        yield self.write_address_busy.acquire()
        for cycle in range(delay):
            yield RisingEdge(self.clock)

        # self.bus.AWADDR <= address
        # self.bus.AWVALID <= 1
        self.bus.awaddr <= address
        self.bus.awvalid <= 1

        while True:
            yield ReadOnly()
            # if self.bus.AWREADY.value:
            if self.bus.awready.value:
                break
            yield RisingEdge(self.clock)
        yield RisingEdge(self.clock)
        # self.bus.AWVALID <= 0
        self.bus.awvalid <= 0
        self.write_address_busy.release()

    @cocotb.coroutine
    def _send_write_data(self, data, delay=0, byte_enable=0xF):
        """
        Send the write address, with optional delay (in clocks)
        """
        yield self.write_data_busy.acquire()
        for cycle in range(delay):
            yield RisingEdge(self.clock)

        # self.bus.WDATA <= data
        # self.bus.WVALID <= 1
        # self.bus.WSTRB <= byte_enable
        self.bus.wdata <= data
        self.bus.wvalid <= 1
        self.bus.wstrb <= byte_enable

        while True:
            yield ReadOnly()
            # if self.bus.WREADY.value:
            if self.bus.wready.value:
                break
            yield RisingEdge(self.clock)
        yield RisingEdge(self.clock)
        # self.bus.WVALID <= 0
        self.bus.wvalid <= 0
        self.write_data_busy.release()

    @cocotb.coroutine
    def write(self, address, value, byte_enable=0xf, address_latency=0,
              data_latency=0, sync=True):
        """
        Write a value to an address.

        Args:
            address (int): The address to write to
            value (int): The data value to write
            byte_enable (int, optional): Which bytes in value to actually write.
                Default is to write all bytes.
            address_latency (int, optional): Delay before setting the address (in clock cycles).
                Default is no delay.
            data_latency (int, optional): Delay before setting the data value (in clock cycles).
                Default is no delay.
            sync (bool, optional): Wait for rising edge on clock initially.
                Defaults to True.
            
        Returns:
            BinaryValue: The write response value
            
        Raises:
            AXIProtocolError: If write response from AXI is not ``OKAY``
        """
        if sync:
            yield RisingEdge(self.clock)

        c_addr = cocotb.fork(self._send_write_address(address,
                                                      delay=address_latency))
        c_data = cocotb.fork(self._send_write_data(value,
                                                   byte_enable=byte_enable,
                                                   delay=data_latency))

        if c_addr:
            yield c_addr.join()
        if c_data:
            yield c_data.join()

        # Wait for the response
        while True:
            yield ReadOnly()
            # if self.bus.BVALID.value and self.bus.BREADY.value:
            if self.bus.bvalid.value and self.bus.bready.value:
                # result = self.bus.BRESP.value
                result = self.bus.bresp.value
                break
            yield RisingEdge(self.clock)

        yield RisingEdge(self.clock)

        if int(result):
            raise AXIProtocolError("Write to address 0x%08x failed with BRESP: %d"
                               % (address, int(result)))

        raise ReturnValue(result)

    @cocotb.coroutine
    def read(self, address, sync=True):
        """
        Read from an address.
        
        Args:
            address (int): The address to read from
            sync (bool, optional): Wait for rising edge on clock initially.
                Defaults to True.
            
        Returns:
            BinaryValue: The read data value
            
        Raises:
            AXIProtocolError: If read response from AXI is not ``OKAY``
        """
        if sync:
            yield RisingEdge(self.clock)

        # self.bus.ARADDR <= address
        # self.bus.ARVALID <= 1
        self.bus.araddr <= address
        self.bus.arvalid <= 1

        while True:
            yield ReadOnly()
            # if self.bus.ARREADY.value:
            if self.bus.arready.value:
                break
            yield RisingEdge(self.clock)

        yield RisingEdge(self.clock)
        # self.bus.ARVALID <= 0
        self.bus.arvalid <= 0

        while True:
            yield ReadOnly()
            # if self.bus.RVALID.value and self.bus.RREADY.value:
            if self.bus.rvalid.value and self.bus.rready.value:
                # data = self.bus.RDATA.value
                # result = self.bus.RRESP.value
                data = self.bus.rdata.value
                result = self.bus.rresp.value
                break
            yield RisingEdge(self.clock)

        if int(result):
            raise AXIProtocolError("Read address 0x%08x failed with RRESP: %d" %
                               (address, int(result)))

        raise ReturnValue(data)

    def __len__(self):
        # return 2**len(self.bus.ARADDR)
        return 2**len(self.bus.araddr)

class AXI4Slave(BusDriver):
    '''
    AXI4 Slave

    Monitors an internal memory and handles read and write requests.
    '''
    _signals = [
        "ARREADY", "ARVALID", "ARADDR",             # Read address channel
        "ARLEN",   "ARSIZE",  "ARBURST", "ARPROT",

        "RREADY",  "RVALID",  "RDATA",   "RLAST",   # Read response channel

        "AWREADY", "AWADDR",  "AWVALID",            # Write address channel
        "AWPROT",  "AWSIZE",  "AWBURST", "AWLEN",

        "WREADY",  "WVALID",  "WDATA",

    ]

    # Not currently supported by this driver
    _optional_signals = [
        "WLAST",   "WSTRB",
        "BVALID",  "BREADY",  "BRESP",   "RRESP",
        "RCOUNT",  "WCOUNT",  "RACOUNT", "WACOUNT",
        "ARLOCK",  "AWLOCK",  "ARCACHE", "AWCACHE",
        "ARQOS",   "AWQOS",   "ARID",    "AWID",
        "BID",     "RID",     "WID"
    ]

    def __init__(self, entity, name, clock, memory, callback=None, event=None,
                 big_endian=False):

        BusDriver.__init__(self, entity, name, clock)
        self.clock = clock

        self.big_endian = big_endian
        self.bus.ARREADY.setimmediatevalue(1)
        self.bus.RVALID.setimmediatevalue(0)
        self.bus.RLAST.setimmediatevalue(0)
        self.bus.AWREADY.setimmediatevalue(1)
        self._memory = memory

        self.write_address_busy = Lock("%s_wabusy" % name)
        self.read_address_busy = Lock("%s_rabusy" % name)
        self.write_data_busy = Lock("%s_wbusy" % name)

        cocotb.fork(self._read_data())
        cocotb.fork(self._write_data())

    def _size_to_bytes_in_beat(self, AxSIZE):
        if AxSIZE < 7:
            return 2 ** AxSIZE
        return None

    @cocotb.coroutine
    def _write_data(self):
        clock_re = RisingEdge(self.clock)

        while True:
            while True:
                self.bus.WREADY <= 0
                yield ReadOnly()
                if self.bus.AWVALID.value:
                    self.bus.WREADY <= 1
                    break
                yield clock_re

            yield ReadOnly()
            _awaddr = int(self.bus.AWADDR)
            _awlen = int(self.bus.AWLEN)
            _awsize = int(self.bus.AWSIZE)
            _awburst = int(self.bus.AWBURST)
            _awprot = int(self.bus.AWPROT)

            burst_length = _awlen + 1
            bytes_in_beat = self._size_to_bytes_in_beat(_awsize)

            word = BinaryValue(n_bits=bytes_in_beat*8, bigEndian=self.big_endian)

            if __debug__:
                self.log.debug(
                    "AWADDR  %d\n" % _awaddr +
                    "AWLEN   %d\n" % _awlen +
                    "AWSIZE  %d\n" % _awsize +
                    "AWBURST %d\n" % _awburst +
                    "BURST_LENGTH %d\n" % burst_length +
                    "Bytes in beat %d\n" % bytes_in_beat)

            burst_count = burst_length

            yield clock_re

            while True:
                if self.bus.WVALID.value:
                    word = self.bus.WDATA.value
                    word.big_endian = self.big_endian
                    _burst_diff = burst_length - burst_count
                    _st = _awaddr + (_burst_diff * bytes_in_beat)  # start
                    _end = _awaddr + ((_burst_diff + 1) * bytes_in_beat)  # end
                    self._memory[_st:_end] = array.array('B', word.get_buff())
                    burst_count -= 1
                    if burst_count == 0:
                        break
                yield clock_re

    @cocotb.coroutine
    def _read_data(self):
        clock_re = RisingEdge(self.clock)

        while True:
            while True:
                yield ReadOnly()
                if self.bus.ARVALID.value:
                    break
                yield clock_re

            yield ReadOnly()
            _araddr = int(self.bus.ARADDR)
            _arlen = int(self.bus.ARLEN)
            _arsize = int(self.bus.ARSIZE)
            _arburst = int(self.bus.ARBURST)
            _arprot = int(self.bus.ARPROT)

            burst_length = _arlen + 1
            bytes_in_beat = self._size_to_bytes_in_beat(_arsize)

            word = BinaryValue(n_bits=bytes_in_beat*8, bigEndian=self.big_endian)

            if __debug__:
                self.log.debug(
                    "ARADDR  %d\n" % _araddr +
                    "ARLEN   %d\n" % _arlen +
                    "ARSIZE  %d\n" % _arsize +
                    "ARBURST %d\n" % _arburst +
                    "BURST_LENGTH %d\n" % burst_length +
                    "Bytes in beat %d\n" % bytes_in_beat)

            burst_count = burst_length

            yield clock_re

            while True:
                self.bus.RVALID <= 1
                yield ReadOnly()
                if self.bus.RREADY.value:
                    _burst_diff = burst_length - burst_count
                    _st = _araddr + (_burst_diff * bytes_in_beat)
                    _end = _araddr + ((_burst_diff + 1) * bytes_in_beat)
                    word.buff = self._memory[_st:_end].tostring()
                    self.bus.RDATA <= word
                    if burst_count == 1:
                        self.bus.RLAST <= 1
                yield clock_re
                burst_count -= 1
                self.bus.RLAST <= 0
                if burst_count == 0:
                    break

class AXI4StreamMaster(BusDriver):

    _signals = ["tvalid", "tready", "tdata"]  # Write data channel
    _optional_signals = ["tlast", "tkeep", "tstrb", "tid", "tdest", "tuser"]

    def __init__(self, entity, name, clock, width=32):
        BusDriver.__init__(self, entity, name, clock)
        #drive default values onto bus
        self.width = width
        self.strobe_width = width / 8
        self.bus.tvalid.setimmediatevalue(0)
        self.bus.tlast.setimmediatevalue(0)
        self.bus.tdata.setimmediatevalue(0)

        if hasattr(self.bus, 'tkeep'):
            self.bus.tkeep.setimmediatevalue(0)

        if hasattr(self.bus, 'tid'):
            self.bus.tid.setimmediatevalue(0)

        if hasattr(self.bus, 'tdest'):
            self.bus.tdest.setimmediatevalue(0)

        if hasattr(self.bus, 'tuser'):
            self.bus.tuser.setimmediatevalue(0)

        self.write_data_busy = Lock("%s_wbusy" % name)

    @cocotb.coroutine
    def write(self, data, byte_enable=-1, keep=1, tid=0, dest=0, user=0):
        """
        Send the write data, with optional delay
        """
        yield self.write_data_busy.acquire()
        self.bus.tvalid <=  0
        self.bus.tlast  <=  0

        if hasattr(self.bus, 'tid'):
            self.bus.tid    <=  tid

        if hasattr(self.bus, 'tdest'):
            self.bus.tdest  <=  dest

        if hasattr(self.bus, 'tuser'):
            self.bus.tuser  <=  user

        if hasattr(self.bus, 'tstrb'):
            self.bus.tstrb  <=  (1 << self.strobe_width) - 1

        if byte_enable == -1:
            byte_enable = (self.width >> 3) - 1

        #Wait for the slave to assert tready
        while True:
            yield ReadOnly()
            if self.bus.tready.value:
                break
            yield RisingEdge(self.clock)

        yield RisingEdge(self.clock)
        #every clock cycle update the data
        for i in range (len(data)):
            self.bus.tvalid <=  1
            self.bus.tdata  <= data[i]
            if i >= len(data) - 1:
                self.bus.tlast  <=  1;
            yield ReadOnly()
            if not self.bus.tready.value:
                while True:
                    yield RisingEdge(self.clock)
                    yield ReadOnly()
                    if self.bus.tready.value:
                        yield RisingEdge(self.clock)
                        break
                continue
            yield RisingEdge(self.clock)

        self.bus.tlast  <= 0;
        self.bus.tvalid <= 0;
        yield RisingEdge(self.clock)
        self.write_data_busy.release()


class AXI4StreamSlave(BusDriver):

    _signals = ["tvalid", "tready", "tdata"]
    _optional_signals = ["tlast", "tkeep", "tstrb", "tid", "tdest", "tuser"]

    def __init__(self, entity, name, clock, width = 32):
        BusDriver.__init__(self, entity, name, clock)
        self.width = width
        self.bus.tready <= 0;
        self.read_data_busy = Lock("%s_wbusy" % name)
        self.data = []

    @cocotb.coroutine
    def read(self, wait_for_valid=True):
        """Read a packe of data from the Axi Ingress stream"""
        yield self.read_data_busy.acquire()
        try:
            # clear the data register and signal we're ready to receive some data
            self.data = []
            self.bus.tready <=  1

            if wait_for_valid:
                while not self.bus.tvalid.value:
                    yield RisingEdge(self.clock)

                # while self.bus.tvalid.value and self.bus.tready.value:
                while True:

                    if self.bus.tvalid.value and self.bus.tready.value:
                        self.data.append(self.bus.tdata.value.integer)

                        if self.bus.tlast.value:
                            self.bus.tready <= 0
                            break
                    
                    yield RisingEdge(self.clock)
                
                
            else:
                self.bus.tready <= 1

                while not self.bus.tlast.value:
                    
                    if self.bus.tvalid.value and self.bus.tready.value:
                        self.data.append(self.bus.tdata.value.integer)
                    
                    yield RisingEdge(self.clock)

                self.bus.tready <= 0
        
        finally:
            self.read_data_busy.release()
                

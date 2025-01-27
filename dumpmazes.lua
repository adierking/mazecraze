-- BizHawk script to generate all possible mazes and dump them to a file
-- This is VERY slow, expect it to take at least 6 hours!

local start_seed = 0x0000
local end_seed = 0xffff

local done = false
local seed = start_seed
local file = assert(io.open("mazes.bin", "wb"))

local function on_gameloop(addr, val, flags)
	mainmemory.write_u16_le(0xbe - 0x80, seed)
end

local function on_maze_ready(addr, val, flags)
	if done then return end
	local dump = mainmemory.read_bytes_as_array(0, 0x3c)
	file:write(string.char(table.unpack(dump)))
	console.log("dumped " .. string.format("%04x", seed))
	if seed == end_seed then
		done = true
		console.log("done!")
	else
		seed = seed + 1
		emu.setregister("PC", 0xfd1e)
	end
end

client.reboot_core()
client.speedmode(1000)
client.invisibleemulation(true)
emu.limitframerate(false)

local event1 = event.on_bus_exec(on_gameloop, 0xfa7e)
local event2 = event.on_bus_exec(on_maze_ready, 0xf87d)

repeat
	emu.frameadvance()
until done

file:close()
event.unregisterbyid(event1)
event.unregisterbyid(event2)
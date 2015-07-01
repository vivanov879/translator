require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

nngraph.setDebug(true)

require 'table_utils'



x = nn.Identity()()
y, z = nn.SplitTable(2,2)(x):split(2)

m = nn.gModule({x}, {z, y})


x = torch.rand(4,4)
y, z = unpack(m:forward(x))
print(x, y, z)




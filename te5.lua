
require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'table_utils'

nngraph.setDebug(true)

x = nn.Identity()()
y = Embedding(4, 10)()
z = nn.Copy()(y)

z = nn.gModule({x}, {z})


l = {'1', 'asdf', '1234'}
s = table.concat(l, ' ')

print(s)
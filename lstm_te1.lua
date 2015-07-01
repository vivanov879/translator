require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

nngraph.setDebug(true)

require 'table_utils'
require 'lstm'


x = nn.Identity()()
y = nn.Identity()()
z = nn.Square()(x)
w = nn.Square()(y)
m = nn.gModule({x, y}, {z, w})


x = nn.Identity()()
y = nn.Identity()()
z, w = m({x, y})
r = nn.CAddTable()({z,w})
n = nn.gModule({x, y}, {z, w, r})

x = torch.rand(2,1)
y = torch.rand(2,1)
z,w,r = unpack(n:forward({x, y}))
print(x)
print(y)
print(z)
print(w)
print(r)



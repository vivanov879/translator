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
h = nn.Identity()()
hh = nn.Copy()(h)
z = nn.Copy()(x)
w = nn.Copy()(y)
m = nn.gModule({x, y, h}, {z, w, hh})


x = nn.Identity()()
y = nn.Identity()()
h = nn.Identity()()
z, w, hh = m({x, y, h}):split(3)
r = nn.CAddTable()({z,w})
n = nn.gModule({x, y, h}, {z, w, r, hh})

x = torch.rand(2,2)
y = torch.rand(2,2)
h = torch.rand(3,3)
z,w,r,h = unpack(n:forward({x, y, h}))
print(x)
print(y)
print(z)
print(w)
print(r)
print(h)

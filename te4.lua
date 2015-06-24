require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'table_utils'

nngraph.setDebug(true)


target = nn.Identity()()
x = nn.Identity()()
y = nn.SoftMax()(x)
z = nn.ClassNLLCriterion()({y, target})
m = nn.gModule({x, target},{z})

x = torch.rand(2,3)
target = torch.rand(2,1)
target[1] = 2
target[2] = 1
z = m:forward({x, target})

print(x)
print(z)




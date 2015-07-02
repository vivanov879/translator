require 'mobdebug'.start()

require 'cutorch'
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

nngraph.setDebug(true)

require 'table_utils'


x = nn.Identity()()
y = nn.Identity()()
z = nn.CAddTable()({x, y})
m = nn.gModule({x,y}, {z})
m1 = m:cuda()

x = nn.Identity()()
y = nn.Identity()()
z = nn.CMulTable()({x, y})
m2 = nn.gModule({x,y}, {z})
m2 = m:cuda()


local params, grad_params = model_utils.combine_all_parameters(m1, m2)






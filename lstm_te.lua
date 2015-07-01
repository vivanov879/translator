require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)

require 'lstm'



opt = {}
opt.rnn_size = 3
opt.n_layers = 2

m = make_lstm_network(opt)
graph.dot(m.fg, 'lstm', 'lstm')

x = torch.rand(5,3)
prev_c = {torch.rand(5,3), torch.rand(5,3)}
prev_h = {torch.rand(5,3), torch.rand(5,3)}
next_x, next_c, next_h = unpack(m:forward({x, prev_c, prev_h}))

a = 1


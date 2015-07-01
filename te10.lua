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
opt.rnn_size = 4
opt.n_layers = 2

m = make_lstm_network(opt)
graph.dot(m.fg, 'lstm', 'lstm')

a = 1


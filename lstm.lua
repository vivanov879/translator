
-- Creates one timestep of one LSTM
function make_lstm_step(opt, x, prev_h, prev_c)
    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_h, next_c
end


function make_lstm_network_old(opt)
  local n_layers = opt.n_layers or 1
  local x = nn.Identity()()
  local prev_s = nn.Identity()()
  local splitted_s = {prev_s:split(2 * n_layers)}
  local next_s = {}
  local inputs = {[0] = x}
  for i = 1, n_layers do 
    local prev_h = splitted_s[2 * i - 1]
    local prev_c = splitted_s[2 * i]
    local next_h, next_c = make_lstm_step(opt, inputs[i - 1], prev_h, prev_c)
    next_s[#next_s + 1] = next_h
    next_s[#next_s + 1] = next_c
    inputs[i] = next_h
  end
  local module = nn.gModule({x, prev_s}, {inputs[n_layers], nn.Identity()(next_s)})
  --module:getParameters():uniform(-0.08, 0.08)
  --module = cuda(module)
  return module
end


function make_lstm_network(opt)
  local n_layers = opt.n_layers or 1
  local x = nn.Identity()()
  local prev_h_unsplit = nn.Identity()()
  local prev_c_unsplit = nn.Identity()()
  local prev_h_split = {prev_h_unsplit:split(n_layers)}
  local prev_c_split = {prev_c_unsplit:split(n_layers)}
  local next_h_unsplit = {}
  local next_c_unsplit = {}
  local inputs = {[0] = x}
  for i = 1, n_layers do 
    local prev_h = prev_h_split[i]:annotate{name='prev_h' .. i}
    local prev_c = prev_c_split[i]:annotate{name='prev_c' .. i}
    local next_h, next_c = make_lstm_step(opt, inputs[i - 1], prev_h, prev_c)
    next_h_unsplit[#next_h_unsplit + 1] = next_h:annotate{name='next_h_unsplit' .. i}
    next_c_unsplit[#next_c_unsplit + 1] = next_c:annotate{name='next_c_unsplit' .. i}
    inputs[i] = next_h
  end
  local module = nn.gModule({x, prev_c_unsplit, prev_h_unsplit}, {inputs[n_layers], nn.Identity()(next_c_unsplit), nn.Identity()(next_h_unsplit)})
  return module
end




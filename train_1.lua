require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'cunn'
require 'gnuplot'
require 'image'
gnuplot.setterm('qt')
local c = require 'trepl.colorize'

local opt = lapp[[
   -s,--save					(default "logs")                       subdirectory to save logs
   -b,--batchSize				(default 1)                            batch size
   -r,--learningRate            (default 1e-3)                         learning rate
   --learningRateDecay          (default 0)                            learning rate decay
   --weightDecay                (default 5e-4)                            weightDecay
   -m,--momentum                (default 0.9)                          momentum
   --epoch_step                 (default 5)                           epoch step
   --max_epoch                  (default 30)                          maximum number of iterations
   --type                       (default cuda)                         cuda or double
]]
print(opt)

-- Tiling functions
function tiling_backward(input, input_squeeze)
    -- Change groundtruth shape to output size
    for idr = 1, input_squeeze:size(3) do
        for idc = 1, input_squeeze:size(4) do
            local tmpr = { (idr-1)*8+1, idr*8 }
            local tmpc = { (idc-1)*8+1, idc*8 }
            input_squeeze[{ {},{},idr,idc }] = torch.reshape(input[{ {},{},tmpr,tmpc }], input_squeeze:size(2), 1)
        end
    end
end

function tiling_forward(input, input_exp)
    -- Tiling: change output to groundtruth shape
    for idr = 1, input:size(3) do
        for idc = 1, input:size(4) do
          local tmpr = { (idr-1)*8+1, idr*8 }
          local tmpc = { (idc-1)*8+1, idc*8 }
	      input_exp[{ {},{},tmpr,tmpc }] = torch.reshape(input[{ {},{},idr,idc }], input_exp:size(2), 8, 8)
        end
    end
end
--------------------------------------------------------


-- Setting parameters
local imH, imW = 480, 640
local outH, outW = imH/4, imW/4
local clsnum = 43

-- Load data
local data = torch.load('./dataset/gtsdb_large_class.t7')
IMG = data[1]
MASK = data[2]
print(#IMG)
print(#MASK)

-- Log
print('Will save at '..opt.save)
paths.mkdir(opt.save)
local testLogger = optim.Logger(paths.concat(opt.save, 'train_info.log'))
testLogger:setNames{'% cost (train set)'}
testLogger.showPlot = false

-- Load model
print(c.red('==> load model'))
--model = dofile('generate_model_1.lua'):cuda()
model = torch.load('./trained_models/model_class_large_iter_30.t7'):cuda()
print(model)
parameters, gradParameters = model:getParameters()

-- Set criterion
print(c.red('==> setting criterion'))
criterion = cudnn.SpatialCrossEntropyCriterion():cuda() 

-- Set optimizer
print(c.red('==> configuring optimizer'))
optimState = {
  learningRate = opt.learningRate,  
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

input = torch.CudaTensor(opt.batchSize, 3, imH, imW):zero()
df_squeeze = torch.CudaTensor(opt.batchSize, 64*clsnum, outH/8, outW/8):zero()
output_exp = torch.CudaTensor(opt.batchSize, clsnum, outH, outW):zero()
target_exp = torch.CudaTensor(opt.batchSize, outH, outW):zero()

function train()
    
  -- Training mod
  model:training()

  local cost = {}

  
  --if opt.type == 'double' then input = input:double()
  --elseif opt.type == 'cuda' then input = input:cuda() target = target:cuda() target = target:cuda() end

  --epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  local trainnum = IMG:size(1)
  shuffle = torch.randperm(trainnum)

  for t = 1, trainnum, opt.batchSize do
    -- Disp progress
    xlua.progress(t, trainnum)

    -- Create mini batch
    local indx = 1
    target_exp:zero()
    output_exp:zero()
    df_squeeze:zero()
  
    for i = t, math.min(t+opt.batchSize-1, trainnum) do
      -- Load new sample
      input[indx] = IMG[shuffle[i]]
      target_exp[indx] = MASK[shuffle[i]]
      indx = indx + 1
    end
    
    -- Create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      
      -- reset gradients
      gradParameters:zero()
      
      -- evaluate function for complete mini batch
      -- estimate f
      
      tiling_forward(model:forward(input), output_exp)
      local f = criterion:forward(output_exp, target_exp)
      
      -- estimate df/dW
      tiling_backward(criterion:backward(output_exp, target_exp), df_squeeze)   
      model:backward(input, df_squeeze)
      
      -- normalize gradients and f(X)
      gradParameters:div(input:size(1))
      
      -- Visualize
      ---[[
      local _, outmask = output_exp[1]:max(1)
      local dd1 = image.toDisplayTensor{input=outmask,
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(64)),
                                        symmetric = true}
      d1 = image.display{image = dd1, win = d1}
      local dd2 = image.toDisplayTensor{input=target_exp[{ 1,{},{} }]:squeeze(),
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(64)),
                                        symmetric = true}
      d2 = image.display{image = dd2, win = d2}
      local dd3 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(96)),
                                        symmetric = true}
      d3 = image.display{image=dd3, win=d3, legend='Layer 1 filters'}
      --]]
      
      table.insert(cost, f)
      
      -- return f and df/dX
      return f, gradParameters
    end -- Local function

    optim.sgd(feval, parameters, optimState)

    --gnuplot.plot(torch.Tensor(cost))

  end -- trainnum
  print("  Mean cost: "..torch.Tensor(cost):mean())
  testLogger:add{"  Mean cost: "..torch.Tensor(cost):mean()}
end -- Train function


for i = 1,  opt.max_epoch do

  print('Epoch '..i)
  testLogger:add{'Epoch '..i}
  train()
  
  if (math.fmod(i, opt.epoch_step) == 0) then
    -- Save model
    torch.save('./trained_models/model_class_large_iter_'..i..'.t7', model)
  end
  
end




        





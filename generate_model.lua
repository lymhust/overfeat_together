require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'
require 'sys'

proto_name = './caffe_model/model_deploy.prototxt'
model_name = './caffe_model/model.caffemodel'

print '==> Loading network'
local model = loadcaffe.load(proto_name, model_name, 'cudnn')

for i = 1, 10 do
  model.modules[#model.modules] = nil -- remove several layers
end
local feaLen1, feaLen2 = 1024, 1024
model:add(cudnn.SpatialConvolution(384, feaLen1, 6, 6, 1, 1, 3, 3, 1))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.500000))

-- Branches
-- Mask
local branch1 = nn.Sequential()
branch1:add(cudnn.SpatialConvolution(feaLen1, feaLen2, 1, 1, 1, 1, 0, 0, 1))
branch1:add(cudnn.ReLU(true))
branch1:add(nn.Dropout(0.500000))
branch1:add(cudnn.SpatialConvolution(feaLen2, 128, 1, 1, 1, 1, 0, 0, 1))

-- BBox
local branch2 = nn.Sequential()
branch2:add(cudnn.SpatialConvolution(feaLen1, feaLen2, 1, 1, 1, 1, 0, 0, 1))
branch2:add(cudnn.ReLU(true))
branch2:add(nn.Dropout(0.500000))
branch2:add(cudnn.SpatialConvolution(feaLen2, 256, 1, 1, 1, 1, 0, 0, 1))
branch2:add(cudnn.Sigmoid())

-- initialization from MSR
---[[
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- Have to do for both backends
  init'cudnn.SpatialConvolution'
end
--MSRinit(model)
MSRinit(branch1)
MSRinit(branch2)
--]]

-- Test model
--[[
-- Concat
local mlp = nn.ConcatTable()
mlp:add(branch1)
mlp:add(branch2)
-- Add branches
model:add(mlp)
model = model:cuda()
local input = torch.CudaTensor(1, 3, 480, 640)
local output = model:forward(input)
print(model)
print(output)

--]]

-- Test speed
--[[
local num = 1000
local input = torch.CudaTensor(1, 3, 480, 640)
local fpsAll = torch.Tensor(num)
for i = 1, num do
    sys.tic()
    local output = model:forward(input:rand(1,3,480,640))
    print('FPS: '..1/sys.toc())
    fpsAll[i] = 1/sys.toc()
end
print('Mean FPS:'..fpsAll:mean())
--]]

-- View kernels
--[[
local d1 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
  padding = 2,
  nrow = math.floor(math.sqrt(96)),
  symmetric = true,
}
image.display{image=d1, legend='Layer 1 filters'}
--]]

return model, branch1, branch2




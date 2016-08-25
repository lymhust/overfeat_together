require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'cunn'
require 'gnuplot'
require 'image'
gnuplot.setterm('qt')
myutils = require 'myutils'
torch.setdefaulttensortype('torch.FloatTensor') 
local c = require 'trepl.colorize'

local opt = lapp[[
   -s,--save					(default "logs")                       subdirectory to save logs
   -b,--batchSize				(default 1)                            batch size
   -r,--learningRate            (default 1e-2)                         learning rate
   --learningRateDecay          (default 0)                            learning rate decay
   --weightDecay                (default 5e-4)                            weightDecay
   -m,--momentum                (default 0.9)                          momentum
   --epoch_step                 (default 1)                           epoch step
   --max_epoch                  (default 2)                          maximum number of iterations
   --type                       (default cuda)                         cuda or double
]]

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

function normalize_global(dataset, mean, std)
  local std = std or dataset:std()
  local mean = mean or dataset:mean()
  dataset:add(-mean)
  dataset:div(std)
  return mean, std
end

function normalize_local(dataset)
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  local normalized_images = norm:forward(dataset)
  dataset:copy(normalized_images)
end
--------------------------------------------------------


-- Setting parameters
local imH, imW = 480, 640
local outH, outW = imH/4, imW/4

-- Load image
local input = image.load('./test_images/00040.ppm')
input = image.scale(input, imW, imH)
local input_ori = input:clone()
input = input:reshape(1, input:size(1), input:size(2), input:size(3))
local output = torch.CudaTensor(1, 43, outH, outW):zero()
local tmp = torch.load('./dataset/MEAN_STD.t7')
local mean, std = tmp[1], tmp[2]

normalize_global(input, mean, std)
normalize_local(input)

-- Load network
model = torch.load('./trained_models/model_class_iter_30.t7'):cuda()
model:evaluate()
tiling_forward(model:forward(input:cuda()), output)
--output = image.flip(output:float(), 3)

output = output:float()
local maxval, mask = output[1]:max(1)
mask = mask[1]
maxval = maxval[1]

local background = torch.Tensor(3, mask:size(1), mask:size(2)):zero()
local graph, indx = myutils.bwlabel(mask)

-- Plot bbox
for i = 2, #indx do
	local tmp = graph:eq(indx[i]):nonzero()
	if (tmp:dim() > 0) then
		local top = tmp[{ {},1 }]:min()
		local bottom = tmp[{ {},1 }]:max()
		local left = tmp[{ {},2 }]:min()
		local right = tmp[{ {},2 }]:max()
		local height = bottom-top+1
		local width = right-left+1
		local cls = mask[tmp[1][1]][tmp[1][2]]
		
		if (height>5 and width>5) then
		    print('Class='..cls)
		    print('Height='..height..', Width='..width)
		    print('top='..top..', bottom='..bottom..', left='..left..', right='..right..'\n')   
		    myutils.drawBox(background, top, bottom, left, right, 1, {math.random(), math.random(), math.random()})
		    d1 = image.display{image = background, win = d1}
		end
	end
end

image.display(input_ori)

        





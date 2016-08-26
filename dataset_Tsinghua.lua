require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
--draw = require 'draw'
local c = require 'trepl.colorize'
torch.setdefaulttensortype('torch.DoubleTensor')
tablex = require 'pl.tablex'

GTFILE = '/media/lym/Work/Code/detection_model/resized/gt.txt'
IMFILE = '/media/lym/Work/Code/detection_model/resized/'

-- Functions
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
  local batch = 10 -- Can be reduced if you experience memory issues
  local dataset_size = dataset:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset:narrow(1,i,local_batch))
    dataset:narrow(1,i,local_batch):copy(normalized_images)
  end
end

function loaddata(folder)

    print(c.red(folder..'...'))
    
    local addr = './'..folder..'/'
    local thnum, thsize = 50, 20
    
    local labnum = torch.Tensor(torch.load(addr..'labelnum.t7'))
    local labhist = labnum:histc(214, 1, 214)
    local numeachcls = labhist[labhist:ge(thnum)]:min()
    local lablist_self = labhist:ge(thnum):nonzero()
    local lablist_all = {}
    local totalnum = numeachcls * lablist_self:size(1)
    
    -- Train
    if (folder == 'train') then
        for i = 1, lablist_self:size(1) do
            lablist_all[i] = lablist_self[i][1]
        end
        torch.save('./lablist_train.t7', lablist_all)
    else
        lablist_all = torch.load('./lablist_train.t7')
    end
    
    local size = 48
    local IMG = torch.Tensor(totalnum, 3, size, size):zero()
    local LABEL = torch.Tensor(totalnum):zero()
    local count = torch.Tensor(#lablist_all):zero()
    local idx = 1
    
    -- Load image info
    local gtruth = torch.load(addr..'gtruth.t7')
    
    for i = 1, #gtruth do
        local tmpgt = gtruth[i]
        local cls = tmpgt[3]
        local img = image.load(addr..tmpgt[1])
     
        if (img:size(2)>thsize and img:size(3)>thsize) then
            local ind = tablex.find(lablist_all, cls)
            if (ind ~= nil) then
                count[ind] = count[ind] + 1 
                if (count[ind] <= numeachcls) then
                    IMG[idx] = image.scale(img, size, size)
                    LABEL[idx] = ind
                    idx = idx + 1
                    print(idx)
                end
            end
        end  
        
        if (idx > totalnum) then print('break') break end  
    end
    
    if (idx < totalnum) then
        IMG = IMG[{ {1,idx-1},{},{},{}}]
        LABEL = LABEL[{ {1,idx-1} }]
    end
    -- Save
    torch.save('./'..folder..'.t7', {IMG, LABEL})
    
end
-------------------------------------------------------------------------------------


local set = {'train','test','other'}

for s = 1, #set do
  -- Load data
  loaddata(set[s])
end--set


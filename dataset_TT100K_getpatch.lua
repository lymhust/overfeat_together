require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
json = require 'json'
draw = require 'draw'
tablex = require 'pl.tablex'
local c = require 'trepl.colorize'
--torch.setdefaulttensortype('torch.FloatTensor')


function loaddata(ids, folder, content)

    print(c.red(folder..' folder number: '..#ids))
    
    local addr = './Tsinghua_small_patch/box/'..folder..'/'
    
    -- Input image size
    local patchid = 1
    
    -- Ground truth file
    local file = io.open(addr..'gtruth.txt', 'w')
    local file_table = {}
    local labelnum = {}

    -- Load img and gtruth
    local imgInfo = content.imgs
    local labelInfo = content.types
    local totalnum = #ids
  
    for i = 1, totalnum do
        
        -- Disp progress
        xlua.progress(i, totalnum)
        
        -- Load image
        local imPath = imgInfo[ids[i]].path
        --local img = image.load(imPath)
        local obj = imgInfo[ids[i]].objects
        
        for j = 1, #obj do
            -- bbox
            local bbox = obj[j].bbox
            local top = math.max(1, bbox.ymin)
            local bottom = math.min(2048, bbox.ymax)
            local left = math.max(1, bbox.xmin)
            local right = math.min(2048, bbox.xmax)
            local w = right-left+1
            local h = bottom-top+1
            
            -- label
            local cls = obj[j].category
            local labnum = tablex.find(labelInfo, cls)
            
            --local patch = img[{ {},{top,bottom},{left,right} }]:clone()
            
            -- Save
            local imgname = patchid..'.jpg'
            --image.save(addr..imgname, patch)
            file:write(imgname..' '..cls..' ')
            file:write(labnum..'\n')
            file_table[patchid] = {imgname, cls, labnum}
            labelnum[patchid] = labnum
            
            patchid = patchid + 1  
            print(patchid)
        end--obj
   
    end--totalnum
    
    file:close()
    torch.save(addr..'gtruth.t7', file_table)
    torch.save(addr..'labelnum.t7', labelnum)

end
-------------------------------------------------------------------------------------

-- Load json
local content = json.load('./annotations.json')
  
local set = {'train','test','other'}
for s = 1, #set do
  -- Load image ids
  local file = assert(io.open(set[s]..'/ids.txt', 'r'))
  local j = 0
  local ids = {}
  for i in file:lines() do
    table.insert(ids, i)
  end
  file:close()
  
  -- Load data
  loaddata(ids, set[s], content)
 
end--set





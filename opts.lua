--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------

   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    'camvid', 'Options: imagenet | cityscapes | camvid')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       1,       'mini-batch size (1 = pure stochastic)')
   cmd:option('-imgH',            0,       'input height')
   cmd:option('-imgW',            0,       'input width')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   cmd:option('-resume',          'none',  'Path to directory containing checkpoint')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.1,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'segnet', 'Options: segnet | deconvnet | headless')
--   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
--   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         14,      'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      end
      -- Default nEpochs=10
      opt.nEpochs = opt.nEpochs == 0 and 10 or opt.nEpochs
      opt.imgH = opt.imgH == 0 and 224 or opt.nEpochs
      opt.imgW = opt.imgW == 0 and 224 or opt.nEpochs
   elseif opt.dataset == 'cityscapes' then
      opt.nEpochs = opt.nEpochs == 0 and 10 or opt.nEpochs
   elseif opt.dataset == 'camvid' then
      opt.nEpochs = opt.nEpochs == 0 and 10 or opt.nEpochs
      opt.imgH = opt.imgH == 0 and 240 or opt.nEpochs
      opt.imgW = opt.imgW == 0 and 320 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end
   print(opt)
   return opt
end

return M

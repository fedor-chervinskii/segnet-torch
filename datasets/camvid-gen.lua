--
--  Copyright (c) 2016, Facebook, Inc.
--  Copyright (c) 2016, Fedor Chervinskii
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arranged with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findImages(dir)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}
   local counter = 0

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end
      counter = counter + 1
      if counter % 100 == 0 then print(counter) end
      local filename = paths.basename(line)
      local path = filename

      table.insert(imagePaths, path)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   return imagePath
end

function M.exec(opt, cacheFile)

   -- find the image path names
   print(" | finding all images")
   local ImagePath = findImages(opt.data .. '/images')
   local LabelPath = findImages(opt.data .. '/labels')

-- dummy split since labels don't matter

   N = ImagePath:size(1)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = ImagePath[{{1,N*0.8},{}}],
         labelPath = LabelPath[{{1,N*0.8},{}}],
      },
      val = {
         imagePath = ImagePath[{{N*0.8+1,N},{}}],
         labelPath = LabelPath[{{N*0.8+1,N},{}}],
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M

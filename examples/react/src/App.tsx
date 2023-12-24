import React, { useEffect, useRef, useState } from 'react'
import './App.css';
import {
  DiffusionPipeline,
  ProgressCallbackPayload,
  setModelCacheDir,
  StableDiffusionPipeline,
  StableDiffusionXLPipeline
} from '@aislamov/diffusers.js'
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { Checkbox, FormControl, InputLabel, MenuItem, Select } from '@mui/material'
import { FormControlLabel } from '@mui/material';
import { BrowserFeatures, hasFp16 } from './components/BrowserFeatures'
import { Tensor } from '@xenova/transformers'
import cv from '@techstark/opencv-js'
import { StableDiffusionControlNetPipeline } from '../../../dist/pipelines/StableDiffusionControlNetPipeline';

const PIXEL_ART_SIZE = 341

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

interface SelectedPipeline {
  name: string
  repo: string
  revision: string
  fp16: boolean
  steps: number
  guidanceScale: number
  pixelArt: boolean
  hasImg2Img: boolean
  hasControlNet: boolean
  hasTimestepCond: boolean
  width: number
  height: number
}

const REPO_PREFIX = "gfodor"

const pipelines = [
   {
     name: 'Pixel Art Detailed Medium Quality (Vega fp16)',
     repo: `${REPO_PREFIX}/segmind-vega-pix-detailed-fp16-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     steps: 60,
     guidanceScale: 4.5,
     pixelArt: true,
   },
   {
     name: 'Pixel Art Flat Color Medium Quality (Vega fp16)',
     repo: `${REPO_PREFIX}/segmind-vega-pix-flat-fp16-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     steps: 60,
     guidanceScale: 4.5,
     pixelArt: true,
   },
   {
     name: '512x512 Turbo (SDXL Turbo fp16)',
     repo: `${REPO_PREFIX}/sdxl-turbo-fp16-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 512,
     height: 512,
     guidanceScale: 0.0001,
     steps: 4,
     pixelArt: false,
   },
   {
     name: '1024x1024 Low Quality Turbo (Vega RT LCM)',
     repo: `${REPO_PREFIX}/segmind-vega-rt-fp16-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     guidanceScale: 0.0001,
     steps: 12,
     pixelArt: false,
   },
   {
     name: '1024x1024 Medium Quality Turbo (SSD1B LCM)',
     repo: `${REPO_PREFIX}/lcm-ssd1b-fp32-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: true,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     guidanceScale: 0.0001,
     steps: 12,
     pixelArt: false,
   },
   /*{
     name: 'Pixel Art Detailed High Quality (SSD1B fp32)',
     repo: `${REPO_PREFIX}/ssd1b-pix-detailed-fp32-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     steps: 60,
     guidanceScale: 4.5,
     pixelArt: true,
   },
   {
     name: 'Pixel Art Flat Color High Quality (SSD1B fp32)',
     repo: `${REPO_PREFIX}/ssd1b-pix-flat-fp32-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     steps: 60,
     guidanceScale: 4,
     pixelArt: true,
   },*/
   {
     name: '1024x1024 Medium-High Quality (Vega Base)',
     repo: `${REPO_PREFIX}/segmind-vega-fp16-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     guidanceScale: 5,
     steps: 60,
     pixelArt: false,
   },
   /*{
     name: '1024x1024 High Quality (SSD1B Base)',
     repo: `${REPO_PREFIX}/ssd1b-fp32-onnx`,
     hasImg2Img: false,
     hasControlNet: false,
     hasTimestepCond: false,
     revision: 'main',
     fp16: true,
     width: 1024,
     height: 1024,
     guidanceScale: 5,
     steps: 60,
     pixelArt: false,
   }*/
]

function App() {
  const [hasF16, setHasF16] = useState<boolean>(false);
  const [selectedPipeline, setSelectedPipeline] = useState<SelectedPipeline|undefined>(pipelines[0]);
  const [modelState, setModelState] = useState<'none'|'loading'|'ready'|'inferencing'>('none');
  //const [prompt, setPrompt] = useState('a raven perched on a branch with dramatic mountains in the background, clouds, mountains, beautiful landscape');
  const [prompt, setPrompt] = useState('A snowy nighttime landscape illuminated by a bright full moon and green auroras in the sky. A snowy owl is perched prominently on a bare birch limb in the foreground, with its distinctive white and gray plumage blending into the wintry scene. In the background, snow-covered coniferous trees.')
  const [negativePrompt, setNegativePrompt] = useState('');
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(5);
  const [seed, setSeed] = useState('');
  const [status, setStatus] = useState('Ready');
  const pipeline = useRef<StableDiffusionXLPipeline|StableDiffusionPipeline|StableDiffusionControlNetPipeline|null>(null);
  const canvasRef = useRef<HTMLCanvasElement|null>(null);
  const [img2img, setImg2Img] = useState(false);
  const [inputImage, setInputImage] = useState<Float32Array>();
  const [strength, setStrength] = useState(0.8);
  const [controlNetImage, setControlNetImage] = useState<Float32Array>();
  const [runVaeOnEachStep, setRunVaeOnEachStep] = useState(false);
  const [maxTokens, setMaxTokens] = useState(0)
  const [totalTokens, setTotalTokens] = useState(0)

  useEffect(() => {
    setModelCacheDir('models')
    hasFp16().then(v => {
      setHasF16(v)
      if (v === false) {
        setSelectedPipeline(pipelines.find(p => p.fp16 === false))
      }
    })
  }, [])

  useEffect(() => {
    setInferenceSteps(selectedPipeline?.steps || 20)
    setGuidanceScale(selectedPipeline?.guidanceScale || 5)
  }, [selectedPipeline])

  useEffect(() => {
    if (pipeline.current) {
      // @ts-ignore
      const maxLength = pipeline.current.tokenizer.model_max_length

      // @ts-ignore
      const tokens = pipeline.current.tokenizer(
        prompt,
        {
          return_tensor: false,
          padding: false,
          // @ts-ignore
          max_length: maxLength,
          return_tensor_dtype: 'int32',
        },
      )

      setMaxTokens(maxLength)
      setTotalTokens(tokens.input_ids.length)
    }
  }, [prompt, selectedPipeline])

  const drawImage = async (image: Tensor) => {
    const canvas = canvasRef.current

    if (canvas) {
      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = selectedPipeline?.width || 1024
      tempCanvas.height = selectedPipeline?.height || 1024
      const tempCtx = tempCanvas.getContext('2d')!
      // @ts-ignore
      const data = await image.toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
      tempCtx.putImageData(data, 0, 0);

      // Now draw the image into the real canvas, with smoothing off, no palettization for now
      const ctx = canvas.getContext('2d')!
      ctx.imageSmoothingEnabled = false
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height)
    }
  }

  const progressCallback = async (info: ProgressCallbackPayload) => {
    if (info.statusText) {
      setStatus(info.statusText)
    }

    if (info.images) {
      // @ts-ignore
      await drawImage(info.images[0])
    }
  }

  const loadModel = async () => {
    if (!selectedPipeline) {
      return
    }
    setModelState('loading')
    try {
      if (pipeline.current) {
        // @ts-ignore
        pipeline.current.release()
      }
      pipeline.current = await DiffusionPipeline.fromPretrained(
        selectedPipeline.repo,
        {
          revision: selectedPipeline?.revision,
          progressCallback
        }
      )
      setModelState('ready')
    } catch (e) {
      alert(e)
      console.error(e)
    }
  }

  /**
   * Extracts the RGB data from an RGBA image array.
   * 
   * @param d RGBA image array.
   * @param normalize Normalizes the image array to [-1,1] if true. Set to true for img2img and false for controlnet.
   * @returns RGB Float32Array.
   */
  function getRgbData(d: Uint8ClampedArray, normalize=true) {
    let rgbData: any = [[], [], []]; // [r, g, b]
    // remove alpha and put into correct shape:
    for(let i = 0; i < d.length; i += 4) {
        let x = (i/4) % 512;
        let y = Math.floor((i/4) / 512)
        if(!rgbData[0][y]) rgbData[0][y] = [];
        if(!rgbData[1][y]) rgbData[1][y] = [];
        if(!rgbData[2][y]) rgbData[2][y] = [];
        rgbData[0][y][x] = normalize ? (d[i+0]/255) * 2 - 1 : (d[i+0]/255);
        rgbData[1][y][x] = normalize ? (d[i+1]/255) * 2 - 1 : (d[i+1]/255);
        rgbData[2][y][x] = normalize ? (d[i+2]/255) * 2 - 1 : (d[i+2]/255);
    }
    rgbData = Float32Array.from(rgbData.flat().flat());
    return rgbData;
  }

  /**
   * Takes an input image and saves it to the corresponding state variable. 
   * The input image can be used either for the controlnet or img2img pipelines
   * which is determined by type.
   * 
   * @param e HTML file upload element.
   * @param type Pipeline of the input image.
   * @returns void
   */
  function uploadImage(e: any, type: 'controlnet'|'img2img') {
    if(!e.target.files[0]) {
      // No image uploaded
      return;
    }

    const uploadedImage = new Image(512, 512); // resize image to 512, 512
    const reader = new FileReader();
    // On file read loadend
    reader.addEventListener('loadend', function(file: any) {
      // On image load
      uploadedImage.addEventListener('load', function() {
        if (type == 'img2img') {
          const imageCanvas = document.createElement('canvas');
          imageCanvas.width = uploadedImage.width;
          imageCanvas.height = uploadedImage.height;
          const imgCtx = imageCanvas.getContext('2d') as CanvasRenderingContext2D;
          imgCtx.drawImage(uploadedImage, 0, 0, uploadedImage.width, uploadedImage.height);
          const imageData = imgCtx.getImageData(0, 0, uploadedImage.width, uploadedImage.height).data;

          const rgb_array = getRgbData(imageData);
          setInputImage(rgb_array);
        }
        else if(type == 'controlnet') {
          // For now only Canny Edge Detection is available
          const cvImg = cv.imread(uploadedImage); // RGBA Image | 4 Channels
          const imgGray = new cv.Mat();
          cv.cvtColor(cvImg, imgGray, cv.COLOR_RGBA2GRAY); // Gray Image | 1 Channel
          const imgCanny = new cv.Mat();
          cv.Canny(imgGray, imgCanny, 100, 200, 3, false); // Canny Image | 1 Channel
          const rgbaCanny = new cv.Mat();
          cv.cvtColor(imgCanny, rgbaCanny, cv.COLOR_GRAY2RGBA, 0); // RGBA Canny Image | 4 Channels

          /**
           * The canny data can be accessed as so:
           * cannyEdges.data -> UInt8Array
           * cannyEdges.data8S -> Int8Array
           * cannyEdges.data16S -> Int16Array
           * cannyEdges.data16U -> UInt16Array
           * cannyEdges.data32F -> Float32Array
           * cannyEdges.data32S -> Int32Array
           * cannyEdges.data64F -> Float64Array
           */

          const rgbCanny = getRgbData(Uint8ClampedArray.from(rgbaCanny.data), false);
          setControlNetImage(rgbCanny);
          cvImg.delete();imgGray.delete();imgCanny.delete();rgbaCanny.delete();
        }
      });
      uploadedImage.src = file.target.result;
    });
    reader.readAsDataURL(e.target.files[0]);
  }

  const runInference = async () => {
    if (!pipeline.current || totalTokens > maxTokens) {
      return
    }
    setModelState('inferencing')

    const images = await pipeline.current.run({
      prompt: (selectedPipeline?.pixelArt ? "bigp1xart, " : "") + prompt,
      negativePrompt: negativePrompt,
      numInferenceSteps: inferenceSteps,
      guidanceScale: guidanceScale,
      seed: seed,
      width: selectedPipeline?.width,
      height: selectedPipeline?.height,
      hasTimestepCond: selectedPipeline?.hasTimestepCond,
      runVaeOnEachStep,
      progressCallback,
      img2imgFlag: img2img,
      inputImage: inputImage,
      strength: strength,
      controlNetImage: controlNetImage
    })
    await drawImage(images[0])
    setModelState('ready')
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline enableColorScheme={true} />
      <Container maxWidth={false} sx={{ maxWidth: 'auto' }}>
        <BrowserFeatures />
        <Stack alignItems={'center'}>
          <p>🎁 Pixel art and image generator by <a href="https://twitter.com/gfodor">@gfodor</a> -- Built with <a href={"https://github.com/dakenf/diffusers.js"} target={"_blank"}>diffusers.js</a> - <a href="https://huggingface.co/gfodor">Huggingface</a></p>
        </Stack>
        <Box sx={{ borderRadius: "12px", bgcolor: '#282c34' }} pt={4} pl={3} pr={3} pb={4}>
          <Grid maxWidth="md" container spacing={2}>
            <Grid item xs={6}>
              <Stack spacing={2}>
                <TextField
                  label="Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setPrompt(e.target.value)}
                  value={prompt}
                />
                {maxTokens > 0 && (<span style={{ color: totalTokens > maxTokens ? "#EC5578" : "rgba(255, 255, 255, 0.5)", fontSize: "0.8em" }}>{totalTokens}/{maxTokens} Tokens</span>)}
                <TextField
                  label="Negative Prompt"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  value={negativePrompt}
                />
                <TextField
                  label="Number of inference steps (Because of PNDM Scheduler, it will be i+1)"
                  variant="standard"
                  type='number'
                  disabled={modelState != 'ready'}
                  onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                  value={inferenceSteps}
                />
                <TextField
                  label="Guidance Scale. Controls how similar the generated image will be to the prompt."
                  variant="standard"
                  type='number'
                  InputProps={{ inputProps: { min: 1, max: 20, step: 0.5 } }}
                  disabled={modelState != 'ready'}
                  onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                  value={guidanceScale}
                />
                <TextField
                  label="Seed (Creates initial random noise)"
                  variant="standard"
                  disabled={modelState != 'ready'}
                  onChange={(e) => setSeed(e.target.value)}
                  value={seed}
                />
                {selectedPipeline?.hasControlNet &&
                  (
                    <>
                      <label htmlFor="upload_controlnet_image">Upload Image for ControlNet Pipeline:</label>
                      <TextField
                        id="upload_controlnet_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={modelState != 'ready'}
                        onChange={(e) => uploadImage(e, "controlnet")}
                      />
                  </>
                )}
                {selectedPipeline?.hasImg2Img &&
                  (
                    <>
                      <FormControlLabel
                        label="Check if you want to use the Img2Img pipeline"
                        control={<Checkbox
                          disabled={modelState != 'ready'}
                          onChange={(e) => setImg2Img(e.target.checked)}
                          checked={img2img}
                        />}
                      />
                      <label htmlFor="upload_image">Upload Image for Img2Img Pipeline:</label>
                      <TextField
                        id="upload_image"
                        inputProps={{accept:"image/*"}}
                        type={"file"}
                        disabled={!img2img}
                        onChange={(e) => uploadImage(e, 'img2img')}
                      />
                      <TextField
                        label="Strength (Noise to add to input image). Value ranges from 0 to 1"
                        variant="standard"
                        type='number'
                        InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
                        disabled={!img2img}
                        onChange={(e) => setStrength(parseFloat(e.target.value))}
                        value={strength}
                      />
                    </>
                )}
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Pipeline</InputLabel>
                    <Select
                      value={selectedPipeline?.name}
                      onChange={e => {
                        setSelectedPipeline(pipelines.find(p => e.target.value === p.name))
                        setModelState('none')
                      }}>
                      {pipelines.map(p => <MenuItem value={p.name} disabled={!hasF16 && p.fp16}>{p.name}</MenuItem>)}
                    </Select>
                </FormControl>
                <Button variant="outlined" onClick={loadModel} disabled={modelState != 'none'}>Load model</Button>
                <Button variant="outlined" onClick={runInference} disabled={modelState != 'ready'}>Run</Button>
                <p>{status}</p>
              </Stack>

            </Grid>
            <Grid item xs={6}>
              <canvas ref={canvasRef}
                width={selectedPipeline?.pixelArt ? PIXEL_ART_SIZE  : selectedPipeline?.width}
                height={selectedPipeline?.pixelArt ? PIXEL_ART_SIZE  : selectedPipeline?.height}
                style={{ 
                  imageRendering: selectedPipeline?.pixelArt ? "pixelated" : "crisp-edges",
                  maxWidth: selectedPipeline?.pixelArt ? `${PIXEL_ART_SIZE}px` : `${selectedPipeline?.width}px`,
                  aspectRatio: "1",
                  width: "50vw",
                  border: '1px dashed #ccc'}} />
                  <button onClick={() => {
                    if (canvasRef.current) {
                      const link = document.createElement('a');
                      link.download = 'image.png';
                      link.href = canvasRef.current.toDataURL()
                      link.click();
                    }
                  }}>Download Image</button>
            </Grid>
          </Grid>
        </Box>
        <Divider/>
      </Container>
    </ThemeProvider>
  );
}

export default App;

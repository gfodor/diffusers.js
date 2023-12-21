import { CLIPTokenizer } from '../tokenizers/CLIPTokenizer'
import { GetModelFileOptions } from '@/hub/common'
import { LCMScheduler } from '@/schedulers/LCMScheduler'
import { PipelineBase } from '@/pipelines/PipelineBase'
import { Session } from '../backends'
import { randomNormalTensor } from '@/util/Tensor'
import { PNDMScheduler, PNDMSchedulerConfig } from '@/schedulers/PNDMScheduler'
import { getModelFile, getModelJSON } from '../hub'
import { Tensor, cat } from '@xenova/transformers'
import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus, sessionRun } from './common'


export interface StableDiffusionXLInput {
  prompt: string
  negativePrompt?: string
  guidanceScale?: number
  seed?: string
  width?: number
  height?: number
  numInferenceSteps: number
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
  img2imgFlag?: boolean
  inputImage?: Float32Array
  strength?: number
}

export class StableDiffusionXLPipeline extends PipelineBase {
  public textEncoder2: Session
  public tokenizer2: CLIPTokenizer
  declare scheduler: PNDMScheduler

  constructor (
    unet: Session,
    vaeDecoder: Session,
    textEncoder: Session,
    textEncoder2: Session,
    tokenizer: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    scheduler: PNDMScheduler,
  ) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.textEncoder = textEncoder
    this.textEncoder2 = textEncoder2
    this.tokenizer = tokenizer
    this.tokenizer2 = tokenizer2
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
  }

  static createScheduler (config: PNDMSchedulerConfig) {
    return new PNDMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    const tokenizer2 = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer_2' })

    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder2 = await loadModel(modelRepoOrPath, 'text_encoder_2/model.onnx', opts)
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionXLPipeline.createScheduler(schedulerConfig)

    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionXLPipeline(unet, vae, textEncoder, textEncoder2, tokenizer, tokenizer2, scheduler)
  }

  async encodePromptXl (prompt: string, tokenizer: CLIPTokenizer, textEncoder: Session, is64: boolean = false) {
    const tokens = tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const inputIds = tokens.input_ids

    const tensor =
      is64
        ? new Tensor('int64', BigInt64Array.from(inputIds.flat().map(x => BigInt(x))), [1, inputIds.length])
        : new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length])

    // @ts-ignore
    return await sessionRun(textEncoder, { input_ids: tensor })
  }

  async getPromptEmbedsXl (prompt: string, negativePrompt: string|undefined) {
    const promptEmbeds = await this.encodePromptXl(prompt, this.tokenizer, this.textEncoder, false)
    let num1HiddenStates = 0

    for (let i = 0; i < 100; i++) {
      if (promptEmbeds[`hidden_states.${i}`] === undefined) {
        break
      }

      num1HiddenStates++
    }

    let posHiddenStates = promptEmbeds[`hidden_states.${num1HiddenStates - 2}`]

    let negHiddenStates

    if (negativePrompt) {
      const negativePromptEmbeds = await this.encodePromptXl(negativePrompt || '', this.tokenizer, this.textEncoder)
      negHiddenStates = negativePromptEmbeds[`hidden_states.${num1HiddenStates - 2}`]
    }

    const promptEmbeds2 = await this.encodePromptXl(prompt, this.tokenizer2, this.textEncoder2, true)

    let num2HiddenStates = 0
    for (let i = 0; i < 100; i++) {
      if (promptEmbeds2[`hidden_states.${i}`] === undefined) {
        break
      }

      num2HiddenStates++
    }

    posHiddenStates = cat([posHiddenStates, promptEmbeds2[`hidden_states.${num2HiddenStates - 2}`]], -1)
    const posTextEmbeds = promptEmbeds2.text_embeds
    let negTextEmbeds

    if (negativePrompt) {
      const negativePromptEmbeds2 = await this.encodePromptXl(negativePrompt || '', this.tokenizer2, this.textEncoder2, true)
      negHiddenStates = cat([negHiddenStates, negativePromptEmbeds2[`hidden_states.${num2HiddenStates - 2}`]], -1)
      negTextEmbeds = negativePromptEmbeds2.text_embeds
    } else {
      negHiddenStates = posHiddenStates.mul(0)
      negTextEmbeds = posTextEmbeds.mul(0)
    }

    return {
      positive: {
        lastHiddenState: posHiddenStates,
        textEmbeds: posTextEmbeds,
      },
      negative: {
        lastHiddenState: negHiddenStates,
        textEmbeds: negTextEmbeds,
      },
    }
  }

  getTimeEmbeds (width: number, height: number) {
    return new Tensor(
      'float32',
      [height, width, 0, 0, height, width],
      [1, 6],
    )
  }

  async run (input: StableDiffusionXLInput) {
    const width = input.width || 1024
    const height = input.height || 1024
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 5
    const seed = input.seed || ''

    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const hasGuidance = guidanceScale >= 1
    const promptEmbeds = await this.getPromptEmbedsXl(input.prompt, hasGuidance ? input.negativePrompt : '')

    const latentShape = [batchSize, 4, width / 8, height / 8]
    let latents = randomNormalTensor(latentShape, undefined, undefined, 'float32', seed) // Normal latents used in Text-to-Image

    let denoised: Tensor
    const timesteps = this.scheduler.timesteps.data
    let humanStep = 1
    let cachedImages: Tensor[]|null = null

    const timeIds = this.getTimeEmbeds(width, height)

    for (const step of timesteps) {
      const timestep = new Tensor(new BigInt64Array([BigInt(step)]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })

      const textNoise = await this.unet.run(
        {
          sample: latents,
          timestep,
          encoder_hidden_states: promptEmbeds.positive.lastHiddenState,
          text_embeds: promptEmbeds.positive.textEmbeds,
          time_ids: timeIds,
        },
      )

      let noisePred

      if (hasGuidance) {
        const uncondNoise = await this.unet.run(
          {
            sample: latents,
            timestep,
            encoder_hidden_states: promptEmbeds.negative.lastHiddenState,
            text_embeds: promptEmbeds.negative.textEmbeds,
            time_ids: timeIds,
          },
        )

        const noisePredUncond = uncondNoise.out_sample
        const noisePredText = textNoise.out_sample
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      } else {
        noisePred = textNoise.out_sample
      }

      const schedulerOutput = this.scheduler.step(
        noisePred,
        step,
        latents,
      )

      latents = schedulerOutput
      denoised = schedulerOutput

      if (this.scheduler instanceof LCMScheduler) {
        latents = schedulerOutput[0]
        denoised = schedulerOutput[1]
      }

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(denoised)
  }

  async release () {
    await super.release()
    return this.textEncoder2?.release()
  }
}

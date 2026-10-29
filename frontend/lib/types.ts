export type ComponentStatus =
  | 'available'
  | 'partial'
  | 'unavailable'
  | 'under_development'
  | 'service_unreachable';

export interface LesionModelStatus {
  key: string;
  code?: string;
  name?: string;
  status: string;
  ready: boolean;
  error: string | null;
  threshold?: number;
  architecture?: string;
  input_size?: number[];
  color?: number[];
  assumptions?: string[];
}

export interface ComponentInfo {
  id: string;
  name: string;
  status: ComponentStatus;
  route: string;
  planned?: string[];
  modelReady?: boolean;
  readyCount?: number;
  totalCount?: number;
  detail?: { ready: boolean; error: string | null; architecture?: string; classNames?: string[] } | null;
  lesions?: Record<string, LesionModelStatus> | null;
}

export interface ComponentsResponse {
  inferenceService: { reachable: boolean; device: string | null };
  components: ComponentInfo[];
  disclaimer: string;
}

export interface ClassProbability {
  grade: number;
  class_name: string;
  probability: number;
}

export interface ClassificationResponse {
  result: {
    grade: number;
    class_name: string;
    confidence: number;
    probabilities: ClassProbability[];
    referable_dr: boolean;
    referable_basis: string;
    referable_probability: number;
    low_confidence: boolean;
    low_confidence_threshold: number;
  };
  image: { width: number; height: number; format: string };
  quality_notes: string[];
  inference_ms: number;
  disclaimer: string;
}

export interface LesionResult {
  code: string;
  name: string;
  color: number[];
  status: 'ok' | 'unavailable' | 'error';
  error?: string;
  mask_png_base64?: string;
  width?: number;
  height?: number;
  threshold?: number;
  positive_pixels?: number;
  positive_fraction?: number;
  max_probability?: number;
  inference_ms?: number;
}

export interface SegmentationResponse {
  image: { width: number; height: number; format: string };
  lesions: Record<string, LesionResult>;
  quality_notes: string[];
  inference_ms: number;
  disclaimer: string;
}

export const LESION_ORDER = ['MA', 'HE', 'EX', 'SE'] as const;

/** Display colours (RGB). Must match the colours the inference service reports. */
export const LESION_COLORS: Record<string, [number, number, number]> = {
  MA: [255, 0, 0],
  HE: [0, 128, 255],
  EX: [255, 215, 0],
  SE: [0, 200, 0],
};

export const LESION_NAMES: Record<string, string> = {
  MA: 'Microaneurysms',
  HE: 'Haemorrhages',
  EX: 'Hard exudates',
  SE: 'Soft exudates',
};

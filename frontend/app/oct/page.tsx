'use client';

import ModalityPage from '@/components/ModalityPage';

export default function OctPage() {
  return (
    <ModalityPage
      title="OCT analysis"
      modality="OCT"
      intro="Optical coherence tomography (OCT) provides cross-sectional images of the retina."
      planned={[
        { title: 'Retinal layer analysis', text: 'Segmentation of retinal layers in OCT B-scans.' },
        { title: 'Fluid / DME analysis', text: 'Detection of retinal fluid and diabetic macular oedema.' },
        { title: 'DR-related classification', text: 'DR-related classification derived from OCT scans.' },
      ]}
    />
  );
}

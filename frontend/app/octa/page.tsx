'use client';

import ModalityPage from '@/components/ModalityPage';

export default function OctaPage() {
  return (
    <ModalityPage
      title="OCTA analysis"
      modality="OCTA"
      intro="Optical coherence tomography angiography (OCTA) visualises retinal microvasculature."
      planned={[
        { title: 'Microvascular analysis', text: 'Quantitative analysis of the retinal microvascular network.' },
        { title: 'Vessel segmentation', text: 'Pixel-level segmentation of the vessels in OCTA en-face images.' },
        { title: 'DR classification', text: 'DR-related classification derived from OCTA scans.' },
      ]}
    />
  );
}

// Integration point for future models.
//
// To connect a real OCTA / OCT model later:
//   1. Add an inference endpoint to the Python service (e.g. POST /v1/octa/analyze).
//   2. Implement a provider here:
//        { analyze: async (file, sniffed, client) => <JSON result> }
//   3. Replace `null` below with that provider and flip `available` in components.js.
// While a provider is `null`, the API answers 501 "model_not_available" and never returns
// invented measurements or predictions.
export const providers = {
  octa: null,
  oct: null,
};

export const UNDER_DEVELOPMENT = {
  octa: {
    id: 'octa',
    name: 'OCTA analysis',
    planned: ['Microvascular analysis', 'Vessel segmentation', 'DR classification'],
  },
  oct: {
    id: 'oct',
    name: 'OCT analysis',
    planned: ['Retinal layer analysis', 'Fluid / DME analysis', 'DR-related classification'],
  },
};

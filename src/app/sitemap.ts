import { MetadataRoute } from 'next';

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: 'https://ml-visualizer.com',
      lastModified: new Date(),
      changeFrequency: 'yearly',
      priority: 1,
    },
    // Add additional routes as you create them
  ];
}

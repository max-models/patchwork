// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import starlightPydocs, { pydocsSidebarGroup } from 'starlight-pydocs';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// The site is served at https://max-models.github.io/patchwork/ by GitHub Pages.
const base = '/patchwork';

// The Python package documented under "API reference" (the directory name in src/).
const pythonPackage = 'patchwork';

export default defineConfig({
	site: 'https://max-models.github.io',
	base,
	markdown: {
		// $...$ and $$...$$ math in Markdown and MDX pages, rendered with KaTeX at build time
		processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }),
	},
	integrations: [
		starlight({
			title: 'Patchwork',
			description: 'Block-structured adaptive meshes backed by mpiarray.',
			customCss: ['katex/dist/katex.min.css', './src/styles/custom.css'],
			components: {
				Header: './src/components/Header.astro',
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/max-models/patchwork' },
			],
			editLink: { baseUrl: 'https://github.com/max-models/patchwork/edit/devel/docs/' },
			lastUpdated: true,
			plugins: [
				starlightPydocs({
					packages: [
						{
							name: pythonPackage,
							label: 'All modules',
							search: ['../src'],
							docstringStyle: 'google',
							members: {
								exclude: [`${pythonPackage}.tests`, `${pythonPackage}.tests.*`],
							},
							sourceLink: {
								host: 'github',
								repo: 'max-models/patchwork',
								ref: 'devel',
								root: '..',
							},
							sidebar: { collapsed: false },
						},
					],
					// The interpreter griffe runs in (set by `make docs-*`); falls back to uvx / python.
					runner: process.env.DOCS_PYTHON ? { python: process.env.DOCS_PYTHON } : undefined,
					inventories: ['python', { url: 'https://numpy.org/doc/stable/objects.inv' }],
				}),
				...(process.env.DOCS_VALIDATE_LINKS === 'false'
					? []
					: [starlightLinksValidator({ errorOnLocalLinks: false, exclude: [`${base}/api/**`, '/api/**'] })]),
			],
			sidebar: [
				{
					label: 'Getting started',
					items: [
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
						{ label: 'Node-centered Poisson', slug: 'getting-started/node-centered' },
					],
				},
				{
					label: 'Tutorials',
					items: [{ autogenerate: { directory: 'tutorials' } }],
				},
				{
					label: 'Development',
					items: [
						{ label: 'Building the docs', slug: 'development/docs' },
						{ label: 'Publishing', slug: 'development/publishing' },
					],
				},
				{
					label: 'API reference',
					items: [pydocsSidebarGroup],
				},
			],
		}),
	],
});

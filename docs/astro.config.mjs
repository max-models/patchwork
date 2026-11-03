// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/patchwork/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'patchwork',
	title: 'Patchwork',
	description: 'Block-structured adaptive meshes backed by mpiarray.',
	pythonPackage: 'patchwork',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, from the mesh to the solvers, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'patchwork.mesh': 'Hierarchy',
			'patchwork.halos': 'Halos and restriction',
			'patchwork.regrid': 'Regridding',
			'patchwork.nodal': 'Nodal Poisson',
			'patchwork.cells': 'Cell operator',
			'patchwork.edges': 'Edge space',
			'patchwork.accumulate': 'Deposit accumulation',
			'patchwork.reflux': 'Refluxing',
			'patchwork.backends': 'Solver backends',
			'patchwork.distributed': 'Distributed solve',
			'patchwork.poisson': 'Legacy cell Poisson',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/patchwork/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Examples',
			items: [{ autogenerate: { directory: 'examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});

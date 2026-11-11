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
			'patchwork.mesh.hierarchy': 'Hierarchy',
			'patchwork.mesh.halos': 'Halos and restriction',
			'patchwork.mesh.regrid': 'Regridding',
			'patchwork.mesh.accumulate': 'Deposit accumulation',
			'patchwork.mesh.reflux': 'Refluxing',
			'patchwork.spaces.boundaries': 'Boundaries and bodies',
			'patchwork.spaces.nodes': 'Node space',
			'patchwork.spaces.edges': 'Edge space',
			'patchwork.spaces.cells': 'Cell space',
			'patchwork.spaces.mass': 'Mass matrices',
			'patchwork.solvers.nodal_poisson': 'Nodal Poisson',
			'patchwork.solvers.cell_operator': 'Cell operator',
			'patchwork.solvers.backends': 'Solver backends',
			'patchwork.solvers.distributed': 'Distributed solve',
			'patchwork.solvers.poisson': 'Legacy cell Poisson',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Guides', link: '/guides/mpi-communication/' },
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
			label: 'Guides',
			items: [{ label: 'MPI communication', slug: 'guides/mpi-communication' }],
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

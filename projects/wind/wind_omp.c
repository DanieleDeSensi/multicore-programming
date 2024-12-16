/*
 * Simplified simulation of air flow in a wind tunnel
 *
 * OpemMP version 
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2020/2021
 *
 * v1.4
 *
 * (c) 2021 Arturo Gonzalez Escribano
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the OpenMP assignment versions */
#include<omp.h>

#define	PRECISION	10000
#define	STEPS		8

/*
 * Student: Comment these macro definitions lines to eliminate modules
 *	Module2: Activate effects of particles in the air pressure
 *	Module3: Activate moving particles
 */
#define MODULE2
#define MODULE3

/* Structure to represent a solid particle in the tunnel surface */
typedef struct {
	unsigned char extra;		// Extra field for student's usage
	int pos_row, pos_col;		// Position in the grid
	int mass;			// Particle mass
	int resistance;			// Resistance to air flow
	int speed_row, speed_col;	// Movement direction and speed
	int old_flow;			// To annotate the flow before applying effects
} Particle;


/* 
 * Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 *
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]

/*
 * Function: Update flow in a matrix position
 * 	This function can be changed and/or optimized by the students
 */
int update_flow( int *flow, int *flow_copy, int *particle_locations, int row, int col, int columns, int skip_particles ) {
	// Skip update in particle positions
	if ( skip_particles && accessMat( particle_locations, row, col ) != 0 ) return 0;

	// Update if border left
	if ( col == 0 ) {
		accessMat( flow, row, col ) = 
			( 
			accessMat( flow_copy, row, col ) + 
			accessMat( flow_copy, row-1, col ) * 2 + 
			accessMat( flow_copy, row-1, col+1 ) 
			) / 4;
	} 
	// Update if border right
	if ( col == columns-1 ) {
		accessMat( flow, row, col ) = 
			( 
			accessMat( flow_copy, row, col ) + 
			accessMat( flow_copy, row-1, col ) * 2 + 
			accessMat( flow_copy, row-1, col-1 ) 
			) / 4;
	}
	// Update in central part
	if ( col > 0 && col < columns-1 ) {
		accessMat( flow, row, col ) = 
			( 
			accessMat( flow_copy, row, col ) + 
			accessMat( flow_copy, row-1, col ) * 2 + 
			accessMat( flow_copy, row-1, col-1 ) +
			accessMat( flow_copy, row-1, col+1 ) 
			) / 5;
	}

	// Return flow variation at this position
	return abs( accessMat( flow_copy, row, col ) - accessMat( flow, row, col ) );
}


/*
 * Function: Move particle
 * 	This function can be changed and/or optimized by the students
 */
void move_particle( int *flow, Particle *particles, int particle, int rows, int columns ) {
	// Compute movement for each step
	int step;
	for( step = 0; step < STEPS; step++ ) {
		// Highly simplified phisical model
		int row = particles[ particle ].pos_row / PRECISION;
		int col = particles[ particle ].pos_col / PRECISION;
		int pressure = accessMat( flow, row-1, col );
		int left, right;
		if ( col == 0 ) left = 0;
		else left = pressure - accessMat( flow, row-1, col-1 );
		if ( col == columns-1 ) right = 0;
		else right = pressure - accessMat( flow, row-1, col+1 );
				
		int flow_row = (int)( (float)pressure / particles[ particle ].mass * PRECISION );
		int flow_col = (int)( (float)(right - left) / particles[ particle ].mass * PRECISION );

		// Speed change	
		particles[ particle ].speed_row =
			( particles[ particle ].speed_row + flow_row ) / 2;
		particles[ particle ].speed_col =
			( particles[ particle ].speed_col + flow_col ) / 2;
	
		// Movement	
		particles[ particle ].pos_row =
			particles[ particle ].pos_row + particles[ particle ].speed_row / STEPS / 2;
		particles[ particle ].pos_col =
			particles[ particle ].pos_col + particles[ particle ].speed_col / STEPS / 2;

		// Control limits
		if ( particles[ particle ].pos_row >= PRECISION * rows )
			particles[ particle ].pos_row = PRECISION * rows - 1;
		if ( particles[ particle ].pos_col < 0 )
			particles[ particle ].pos_col = 0;
		if ( particles[ particle ].pos_col >= PRECISION * columns )
			particles[ particle ].pos_col = PRECISION * columns - 1;
	}
}


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, int *flow, int num_particles, int *particle_locations, int max_var ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty 
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;
	printf("Iteration: %d, max_var: %f\n", 
		iteration, 
		(float)max_var / PRECISION
		);

	printf("  +");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		if ( i % STEPS == iteration % STEPS )
			printf("->|");
		else
			printf("  |");

		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( flow, i, j )  >= 10 * PRECISION ) symbol = '*';
			else if ( accessMat( flow, i, j ) >= 1 * PRECISION ) symbol = '0' + accessMat( flow, i, j ) / PRECISION;
			else if ( accessMat( flow, i, j ) >= 0.5 * PRECISION ) symbol = '+';
			else if ( accessMat( flow, i, j ) > 0 ) symbol = '.';
			else symbol = ' ';

			if ( accessMat( particle_locations, i, j ) > 0 ) printf("[%c]", symbol );
			else printf(" %c ", symbol );
		}
		printf("|\n");
	}
	printf("  +");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n\n");
}
#endif

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <maxIter> <threshold> <inlet_pos> <inlet_size> <fixed_particles_pos> <fixed_particles_size> <fixed_particles_density> <moving_particles_pos> <moving_particles_size> <moving_particles_density> <short_rnd1> <short_rnd2> <short_rnd3> [ <fixed_row> <fixed_col> <fixed_resistance> ... ]\n");
	fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j;

	// Simulation data
	int max_iter;			// Maximum number of simulation steps
	int var_threshold;		// Threshold of variability to continue the simulation
	int rows, columns;		// Cultivation area sizes

	int *flow;			// Wind tunnel air-flow 
	int *flow_copy;			// Wind tunnel air-flow (ancillary copy)
	int *particle_locations;	// To quickly locate places with particles

	int inlet_pos;			// First position of the inlet
	int inlet_size;			// Inlet size
	int particles_f_band_pos;	// First position of the band where fixed particles start
	int particles_f_band_size;	// Size of the band where fixed particles start
	int particles_m_band_pos;	// First position of the band where moving particles start
	int particles_m_band_size;	// Size of the band where moving particles start
	float particles_f_density;	// Density of starting fixed particles
	float particles_m_density;	// Density of starting moving particles

	unsigned short random_seq[3];		// Status of the random sequence

	int		num_particles;		// Number of particles
	Particle	*particles;		// List to store cells information

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 16) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read simulation area sizes, maximum number of iterations and threshold */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	max_iter = atoi( argv[3] );
	var_threshold = (int)(atof( argv[4] ) * PRECISION);

	/* 1.3. Read inlet data and band of moving particles data */
	inlet_pos = atoi( argv[5] );
	inlet_size = atoi( argv[6] );
	particles_f_band_pos = atoi( argv[7] );
	particles_f_band_size = atoi( argv[8] );
	particles_f_density = atof( argv[9] );
	particles_m_band_pos = atoi( argv[10] );
	particles_m_band_size = atoi( argv[11] );
	particles_m_density = atof( argv[12] );

	/* 1.4. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		random_seq[i] = (unsigned short)atoi( argv[13+i] );
	}

	/* 1.5. Allocate particles */
	num_particles = 0;
	// Check correct number of parameters for fixed particles
	if (argc > 16 ) {
		if ( (argc - 16) % 3 != 0 ) {
			fprintf(stderr, "-- Error in number of fixed position particles\n\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		// Get number of fixed particles
		num_particles = (argc - 16) / 3;
	}
	// Add number of fixed and moving particles in the bands
	int num_particles_f_band = (int)( particles_f_band_size * columns * particles_f_density );
	int num_particles_m_band = (int)( particles_m_band_size * columns * particles_m_density );
	num_particles += num_particles_f_band;
	num_particles += num_particles_m_band;

	// Allocate space for particles
	if ( num_particles > 0 ) {
		particles = (Particle *)malloc( num_particles * sizeof( Particle ) );
		if ( particles == NULL ) {
			fprintf(stderr,"-- Error allocating particles structure for size: %d\n", num_particles );
			exit( EXIT_FAILURE );
		}
	}
	else particles = NULL;

	/* 1.6.1. Read fixed particles */
	int particle = 0;
	if (argc > 16 ) {
		int fixed_particles = (argc - 16) / 3;
		for (particle = 0; particle < fixed_particles; particle++) {
			particles[ particle ].pos_row = atoi( argv[ 16 + particle*3 ] ) * PRECISION;
			particles[ particle ].pos_col = atoi( argv[ 17 + particle*3 ] ) * PRECISION;
			particles[ particle ].mass = 0;
			particles[ particle ].resistance = (int)( atof( argv[ 18 + particle*3 ] ) * PRECISION);
			particles[ particle ].speed_row = 0;
			particles[ particle ].speed_col = 0;
		}
	}
	/* 1.6.2. Generate fixed particles in the band */
	for ( ; particle < num_particles-num_particles_m_band; particle++) {
		particles[ particle ].pos_row = (int)( PRECISION * ( particles_f_band_pos + particles_f_band_size * erand48( random_seq ) ) );
		particles[ particle ].pos_col = (int)( PRECISION * columns * erand48( random_seq ) );
		particles[ particle ].mass = 0;
		particles[ particle ].resistance = (int)( PRECISION * erand48( random_seq ) );
		particles[ particle ].speed_row = 0;
		particles[ particle ].speed_col = 0;
	}

	/* 1.7. Generate moving particles in the band */
	for ( ; particle < num_particles; particle++) {
		particles[ particle ].pos_row = (int)( PRECISION * ( particles_m_band_pos + particles_m_band_size * erand48( random_seq ) ) );
		particles[ particle ].pos_col = (int)( PRECISION * columns * erand48( random_seq ) );
		particles[ particle ].mass = (int)( PRECISION * ( 1 + 5 * erand48( random_seq ) ) );
		particles[ particle ].resistance = (int)( PRECISION * erand48( random_seq ) );
		particles[ particle ].speed_row = 0;
		particles[ particle ].speed_col = 0;
	}

#ifdef DEBUG
	// 1.8. Print arguments 
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d, threshold: %f\n", rows, columns, max_iter, (float)var_threshold/PRECISION );
	printf("Arguments, Inlet: %d, %d  Band of fixed particles: %d, %d, %f  Band of moving particles: %d, %d, %f\n", inlet_pos, inlet_size, particles_f_band_pos, particles_f_band_size, particles_f_density, particles_m_band_pos, particles_m_band_size, particles_m_density );
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
	printf("Particles: %d\n", num_particles );
	for (int particle=0; particle<num_particles; particle++) {
		printf("Particle[%d] = { %d, %d, %d, %d, %d, %d }\n",
				particle,
				particles[particle].pos_row, 
				particles[particle].pos_col, 
				particles[particle].mass, 
				particles[particle].resistance, 
				particles[particle].speed_row, 
				particles[particle].speed_col
				);
	}
	printf("\n");
#endif // DEBUG


	/* 2. Start global timer */
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

	/* 3. Initialization */
	flow = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	flow_copy = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	particle_locations = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );

	if ( flow == NULL || flow_copy == NULL || particle_locations == NULL ) {
		fprintf(stderr,"-- Error allocating culture structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}

	for( i=0; i<rows; i++ ) {
		for( j=0; j<columns; j++ ) {
			accessMat( flow, i, j ) = 0;
			accessMat( flow_copy, i, j ) = 0;
			accessMat( particle_locations, i, j ) = 0;
		}
	}


	/* 4. Simulation */
	int max_var = INT_MAX;
	int iter;
	for( iter=1; iter<=max_iter && max_var > var_threshold; iter++) {

		// 4.1. Change inlet values each STEP iterations
		if ( iter % STEPS == 1 ) {
			for ( j=inlet_pos; j<inlet_pos+inlet_size; j++ ) {
				// 4.1.1. Change the fans phase
				double phase = iter / STEPS * ( M_PI / 4 );
				double phase_step = M_PI / 2 / inlet_size;
				double pressure_level = 9 + 2 * sin( phase + (j-inlet_pos) * phase_step );
	
				// 4.1.2. Add some random noise
				double noise = 0.5 - erand48( random_seq );

				// 4.1.3. Store level in the first row of the ancillary structure
				accessMat( flow, 0, j ) = (int)(PRECISION * (pressure_level + noise));
			}
		} // End inlet update


#ifdef MODULE2
#ifdef MODULE3
		// 4.2. Particles movement each STEPS iterations
		if ( iter % STEPS == 1 ) {
			// Clean particle positions
			for( i=0; i<=iter && i<rows; i++ ) 
				for( j=0; j<columns; j++ )
					accessMat( particle_locations, i, j ) = 0;

			int particle;
			for( particle = 0; particle < num_particles; particle++ ) {
				int mass = particles[ particle ].mass;
				// Fixed particles
				if ( mass == 0 ) continue;
				// Movable particles
				move_particle( flow, particles, particle, rows, columns );
			} 

			// Annotate position
			for( particle = 0; particle < num_particles; particle++ ) {
				accessMat( particle_locations, 
					particles[ particle ].pos_row / PRECISION,
					particles[ particle ].pos_col / PRECISION ) += 1;
			}
		} // End particles movements
#endif // MODULE3

		// 4.3. Effects due to particles each STEPS iterations
		if ( iter % STEPS == 1 ) {
			int particle;
			for( particle = 0; particle < num_particles; particle++ ) {
				int row = particles[ particle ].pos_row / PRECISION;
				int col = particles[ particle ].pos_col / PRECISION;

				update_flow( flow, flow_copy, particle_locations, row, col, columns, 0 );
				particles[ particle ].old_flow = accessMat( flow, row, col );
			}
			for( particle = 0; particle < num_particles; particle++ ) {
				int row = particles[ particle ].pos_row / PRECISION;
				int col = particles[ particle ].pos_col / PRECISION;
				int resistance = particles[ particle ].resistance;

				int back = (int)( (long)particles[ particle ].old_flow * resistance / PRECISION ) / accessMat( particle_locations, row, col );
				accessMat( flow, row, col ) -= back;
			
				accessMat( flow, row-1, col ) += back / 2;

				if ( col > 0 )
					accessMat( flow, row-1, col-1 ) += back / 4;
				else
					accessMat( flow, row-1, col ) += back / 4;
				if ( col < columns-1 )
					accessMat( flow, row-1, col+1 ) += back / 4;
				else
					accessMat( flow, row-1, col ) += back / 4;
			}
		} // End effects
#endif // MODULE2


		// 4.4. Copy data in the ancillary structure
		for( i=0; i<iter && i<rows; i++ ) 
			for( j=0; j<columns; j++ )
				accessMat( flow_copy, i, j ) = accessMat( flow, i, j );


		// 4.5. Propagation stage
		// 4.5.1. Initialize data to detect maximum variability
		if ( iter % STEPS == 1 ) max_var = 0;

		// 4.5.2. Execute propagation on the wave fronts
		int wave_front = iter % STEPS;
		if ( wave_front == 0 ) wave_front = STEPS;
		int wave;
		for( wave = wave_front; wave < rows; wave += STEPS) {
			if ( wave > iter ) continue;
			int col;
			for( col=0; col<columns; col++) {
				int var = update_flow( flow, flow_copy, particle_locations, wave, col, columns, 1 );
				if ( var > max_var ) {
					max_var = var;
				}
			}
		} // End propagation
		
#ifdef DEBUG
		// 4.7. DEBUG: Print the current state of the simulation at the end of each iteration 
		print_status( iter, rows, columns, flow, num_particles, particle_locations, max_var );
#endif

	} // End iterations

	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global timer */
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 6.2. Results: Statistics */
	printf("Result: %d, %d", 
		iter-1, 
		max_var
		);
	int res_row = ( iter-1 < rows-1 ) ? iter-1 : rows-1;
	int ind;
	for ( ind=0; ind<6; ind++ )
		printf(", %d", accessMat( flow, STEPS-1, ind * columns/6 ) );
	for ( ind=0; ind<6; ind++ )
		printf(", %d", accessMat( flow, res_row/2, ind * columns/6 ) );
	for ( ind=0; ind<6; ind++ )
		printf(", %d", accessMat( flow, res_row, ind * columns/6 ) );
	printf("\n");

	/* 7. Free resources */	
	free( flow );
	free( flow_copy );
	free( particle_locations );
	free( particles );

	/* 8. End */
	return 0;
}
